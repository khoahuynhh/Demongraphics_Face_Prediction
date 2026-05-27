"""
Face Attributes Prediction Model and Training Script
Author: Khoa Thieu Gia
"""

import os
import numpy as np
import math
import random
import sys
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms, models
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
from PIL import Image
from datetime import datetime


# --------------------
class Logger(object):
    def __init__(self, filename="training.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


# Set seeds for consistency
def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)


def seed_worker(worker_id):
    """
    Seed for each worker DataLoader
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class Config:
    # General configuration
    DATA_ROOT = "./data"  # Path to data
    TRAIN_LIST = "./data/train_new.txt"  # Training list file
    VAL_LIST = "./data/val.txt"  # Validation list file
    OUTPUT_DIR = "./outputs/resnet50"  # Output directory for the model

    # Model parameters
    BACKBONE = "resnet50"  # Backbone: resnet18, resnet34, resnet50, resnet101
    USE_PRETRAINED = (
        False  # True only for the external ResNet staged fine-tuning branch
    )
    EMBEDDING_SIZE = 512  # Embedding vector size
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Training parameters
    BATCH_SIZE = 64
    NUM_WORKERS = 4
    NUM_EPOCHS = 200
    LEARNING_RATE = 1e-4
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-4

    # Image configuration
    IMG_SIZE = 224  # Input image size

    # Evaluation configuration
    EVAL_BATCH_SIZE = 128
    EVAL_FREQUENCY = 1  # Evaluation frequency (after how many epochs)
    EARLY_STOPPING_PATIENCE = 10  # Early stopping patience


GENDER_NAMES = ["male", "female"]
EMOTION_NAMES = ["anger", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


class EarlyStopping:
    """
    Early stop during training based on a validation metric.
    mode='min' for val_loss, mode='max' for val_acc/val_auc.
    """

    def __init__(self, patience=8, min_delta=1e-4, mode="min", warmup_epochs=0):
        assert mode in ["min", "max"]
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = mode
        self.warmup_epochs = int(warmup_epochs)

        self.best = None
        self.bad_count = 0
        self.epoch = 0

    def _is_better(self, value, best):
        if self.mode == "min":
            return (best - value) > self.min_delta
        return (value - best) > self.min_delta

    def step(self, value: float) -> bool:
        """Call once per epoch after validation. Return True if should stop."""
        self.epoch += 1

        # ignore nan/inf
        if value is None or (
            isinstance(value, float) and (math.isnan(value) or math.isinf(value))
        ):
            self.bad_count += 1
            return self.bad_count >= self.patience

        # warmup
        if self.epoch <= self.warmup_epochs:
            self.best = (
                value
                if self.best is None
                else (
                    min(self.best, value)
                    if self.mode == "min"
                    else max(self.best, value)
                )
            )
            return False

        # init best
        if self.best is None:
            self.best = value
            return False

        # normal
        if self._is_better(value, self.best):
            self.best = value
            self.bad_count = 0
        else:
            self.bad_count += 1

        return self.bad_count >= self.patience


# --------------------
# PART 1: DATA
# --------------------


class FaceAttrDataset(Dataset):
    def __init__(self, data_root, list_file, transform=None):
        self.data_root = data_root
        self.transform = transform
        self.target_size = Config.IMG_SIZE
        self.items = []

        with open(list_file, "r", encoding="utf-8") as f:
            for i, raw in enumerate(f):
                if i == 0:
                    continue
                raw = raw.strip()
                if not raw or raw.startswith("#"):
                    continue

                parts = raw.split()
                if len(parts) != 4:
                    raise ValueError(f"Invalid line in {list_file}: {raw}")

                p, age, gender, emo = parts
                self.items.append((p, float(age), int(gender), int(emo)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rel, age, gender, emo = self.items[idx]
        img_path = os.path.join(self.data_root, rel)

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img_rgb)

        if self.transform is not None:
            img = self.transform(img)

        # Default DataLoader collation can fail on tensors backed by
        # non-resizable storage from image transforms. Return a regular
        # contiguous clone so worker collation is stable.
        if isinstance(img, torch.Tensor):
            img = img.contiguous().clone()

        return (
            img,
            torch.tensor(age, dtype=torch.float32),
            torch.tensor(gender, dtype=torch.long),
            torch.tensor(emo, dtype=torch.long),
        )


def get_train_transform():
    return transforms.Compose(
        [
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_val_transform():
    return transforms.Compose(
        [
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


# --------------------
# PART 2: MODEL
# --------------------


def freeze_layers(model, freeze_until="layer3", verbose=True):
    resnet = model.backbone.backbone

    stages = ["stem", "layer1", "layer2", "layer3", "layer4"]
    freeze_set = set()

    if freeze_until == "all":
        freeze_set = set(stages)
    elif freeze_until in stages:
        idx = stages.index(freeze_until)
        freeze_set = set(stages[: idx + 1])
    elif freeze_until == "none":
        freeze_set = set()
    else:
        raise ValueError(f"Unknown freeze_until: {freeze_until}")

    # unfreeze all
    for p in model.parameters():
        p.requires_grad = True

    # freeze selected stages
    for name in freeze_set:
        if name == "stem":
            modules = [resnet.conv1, resnet.bn1]
        else:
            modules = [getattr(resnet, name)]

        for m in modules:
            for p in m.parameters():
                p.requires_grad = False

    if verbose:
        print(f"[freeze] Frozen stages: {sorted(freeze_set)}")


def freeze_bn(module):
    for m in module.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False


class RecorderMeter:
    pass


class AverageMeter:
    pass


def numpy_to_torch(state_dict, device="cpu"):
    new_state = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            new_state[k] = v.to(device)
        elif isinstance(v, np.ndarray):
            new_state[k] = torch.from_numpy(v).to(device)
        else:
            raise TypeError(f"Key {k}: unsupported type {type(v)}")
    return new_state


class FaceBackbone(nn.Module):
    """
    Face attribute backbone based on torchvision ResNet variants.
    """

    def __init__(
        self, backbone_name="resnet50", pretrained: str | None = None, device="cpu"
    ):
        super(FaceBackbone, self).__init__()

        # Load the backbone model from torchvision
        if backbone_name == "resnet18":
            self.backbone = models.resnet18(weights="DEFAULT")
            feature_dim = 512
        elif backbone_name == "resnet34":
            self.backbone = models.resnet34(weights="DEFAULT")
            feature_dim = 512
        elif backbone_name == "resnet50":
            self.backbone = models.resnet50(weights="DEFAULT")
            feature_dim = 2048
        elif backbone_name == "resnet101":
            self.backbone = models.resnet101(weights="DEFAULT")
            feature_dim = 2048
        elif backbone_name == "resnet50_pretrained":
            self.backbone = models.resnet50(weights=None)
            feature_dim = 2048
            if pretrained is not None:
                if pretrained.endswith(".pth") or pretrained.endswith(".pt"):
                    ckpt = torch.load(
                        pretrained, map_location=device, weights_only=False
                    )
                elif pretrained.endswith(".pkl"):
                    with open(pretrained, "rb") as f:
                        ckpt = pickle.load(f)
                state = (
                    ckpt["state_dict"]
                    if isinstance(ckpt, dict) and "state_dict" in ckpt
                    else ckpt
                )

                # strip common prefix
                def strip_prefix(sd, prefix):
                    return {
                        (k[len(prefix) :] if k.startswith(prefix) else k): v
                        for k, v in sd.items()
                    }

                state = strip_prefix(state, "module.")
                state = strip_prefix(state, "backbone.")
                state = strip_prefix(state, "model.")

                # remove fc layer weights
                state = {k: v for k, v in state.items() if not k.startswith("fc.")}

                # convert numpy to torch tensors if needed
                state = numpy_to_torch(state, device=device)

                # load into torchvision resnet
                missing, unexpected = self.backbone.load_state_dict(state, strict=False)
                print(f"[FaceBackbone] Loaded from {pretrained}")
                print("  Missing (ex):", missing[:5])
                print("  Unexpected (ex):", unexpected[:5])
            else:
                print("[INFO] No external pretrained backbone loaded.")
        else:
            raise ValueError(f"Backbone {backbone_name} is not supported")

        resnet = self.backbone  # torchvision resnet

        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

        # Remember output feature dimension
        self.feature_dim = feature_dim

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        # Discard last fully connected layer
        # x = self.se(x)
        x = torch.flatten(x, 1)  # Flatten tensor
        return x


class FaceAttrModel(nn.Module):
    def __init__(self, backbone_name="resnet50", pretrained=None, emo_classes=7):
        super().__init__()
        self.backbone = FaceBackbone(backbone_name=backbone_name, pretrained=pretrained)

        feature_dim = (
            self.backbone.feature_dim
        )  # 2048 for resnet50, 512 for resnet18/34

        self.bottleneck = nn.Sequential(
            nn.Linear(feature_dim, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True)
        )

        # Age regression (B,) -> predicts numeric age
        self.age_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        # Gender classification (B,2)
        self.gender_head = nn.Linear(512, 2)

        # Emotion classification (B,K)
        self.emo_head = nn.Linear(512, emo_classes)

        self.log_vars = nn.Parameter(torch.zeros(3))

    def forward(self, x):
        pooled = self.backbone(x)  # (B, feature_dim)
        feat = self.bottleneck(pooled)

        age = self.age_head(feat).squeeze(1) * 100  # (B,)
        gender_logits = self.gender_head(feat)  # (B,2)
        emotion_logits = self.emo_head(feat)  # (B,K)

        return {
            "feat": feat,
            "age": age,
            "gender": gender_logits,
            "emotion": emotion_logits,
            "log_vars": self.log_vars,
        }


# --------------------
# PART 3: TRAINING
# --------------------

# Fixed weights

# def train_one_epoch_attr(model, loader, optimizer, device, weights=(1.0, 1.0, 1.0)):
#     model.train()

#     age_loss_fn = nn.SmoothL1Loss(beta=1.0)
#     gender_loss_fn = nn.CrossEntropyLoss()
#     emo_loss_fn = nn.CrossEntropyLoss()
#     w_age, w_gender, w_emo = weights

#     total_loss_sum = 0.0
#     total_samples = 0

#     # metrics (masked)
#     age_mae_sum, age_cnt = 0.0, 0
#     gender_ok, gender_cnt = 0, 0
#     emo_ok, emo_cnt = 0, 0

#     for imgs, age, gender, emo in loader:
#         imgs = imgs.to(device, non_blocking=True)
#         age = age.to(device, non_blocking=True)  # float (B,)
#         gender = gender.to(device, non_blocking=True)  # long (B,)
#         emo = emo.to(device, non_blocking=True)  # long (B,)

#         out = model(imgs)

#         loss = torch.tensor(0.0, device=device)

#         # Age
#         m_age = age != -1
#         if m_age.any():
#             loss_age = age_loss_fn(out["age"][m_age], age[m_age])
#             loss = loss + w_age * loss_age
#             age_mae_sum += (out["age"][m_age] - age[m_age]).abs().sum().item()
#             age_cnt += int(m_age.sum().item())

#         # Gender
#         m_gender = gender != -1
#         if m_gender.any():
#             loss_gender = gender_loss_fn(out["gender"][m_gender], gender[m_gender])
#             loss = loss + w_gender * loss_gender
#             pred_g = out["gender"][m_gender].argmax(dim=1)
#             gender_ok += (pred_g == gender[m_gender]).sum().item()
#             gender_cnt += int(m_gender.sum().item())

#         # Emotion
#         m_emo = emo != -1
#         if m_emo.any():
#             loss_emo = emo_loss_fn(out["emotion"][m_emo], emo[m_emo])
#             loss = loss + w_emo * loss_emo
#             pred_e = out["emotion"][m_emo].argmax(dim=1)
#             emo_ok += (pred_e == emo[m_emo]).sum().item()
#             emo_cnt += int(m_emo.sum().item())

#         optimizer.zero_grad(set_to_none=True)
#         loss.backward()
#         optimizer.step()

#         bs = imgs.size(0)
#         total_loss_sum += float(loss.item()) * bs
#         total_samples += bs

#     return {
#         "loss": total_loss_sum / max(total_samples, 1),
#         "age_mae": (age_mae_sum / age_cnt) if age_cnt > 0 else None,
#         "gender_acc": (100.0 * gender_ok / gender_cnt) if gender_cnt > 0 else None,
#         "emo_acc": (100.0 * emo_ok / emo_cnt) if emo_cnt > 0 else None,
#         "counts": {"age": age_cnt, "gender": gender_cnt, "emotion": emo_cnt},
#     }


# @torch.no_grad()
# def eval_one_epoch_attr(model, dataloader, device, weights=(1.0, 1.0, 1.0)):
#     model.eval()

#     age_loss_fn = nn.SmoothL1Loss(beta=1.0)
#     gender_loss_fn = nn.CrossEntropyLoss()
#     emo_loss_fn = nn.CrossEntropyLoss()
#     w_age, w_gender, w_emo = weights

#     total_loss = 0.0
#     n_batches = 0

#     # metrics accumulators (masked)
#     age_mae_sum, age_cnt = 0.0, 0
#     gender_ok, gender_cnt = 0, 0
#     emo_ok, emo_cnt = 0, 0

#     for inputs, age, gender, emo in dataloader:
#         inputs = inputs.to(device, non_blocking=True)
#         age = age.to(device, non_blocking=True)  # float (B,)
#         gender = gender.to(device, non_blocking=True)  # long (B,)
#         emo = emo.to(device, non_blocking=True)  # long (B,)

#         out = model(inputs)  # dict: age, gender, emotion

#         loss = 0.0

#         # Age (mask age != -1)
#         m_age = age != -1
#         if m_age.any():
#             loss_age = age_loss_fn(out["age"][m_age], age[m_age])
#             loss += w_age * loss_age
#             age_mae_sum += (out["age"][m_age] - age[m_age]).abs().sum().item()
#             age_cnt += int(m_age.sum().item())

#         # Gender (mask gender != -1)
#         m_gender = gender != -1
#         if m_gender.any():
#             loss_gender = gender_loss_fn(out["gender"][m_gender], gender[m_gender])
#             loss += w_gender * loss_gender
#             pred_g = out["gender"][m_gender].argmax(dim=1)
#             gender_ok += (pred_g == gender[m_gender]).sum().item()
#             gender_cnt += int(m_gender.sum().item())

#         # Emotion (mask emo != -1)
#         m_emo = emo != -1
#         if m_emo.any():
#             loss_emo = emo_loss_fn(out["emotion"][m_emo], emo[m_emo])
#             loss += w_emo * loss_emo
#             pred_e = out["emotion"][m_emo].argmax(dim=1)
#             emo_ok += (pred_e == emo[m_emo]).sum().item()
#             emo_cnt += int(m_emo.sum().item())

#         total_loss += float(loss)
#         n_batches += 1

#     avg_loss = total_loss / max(n_batches, 1)

#     return {
#         "loss": avg_loss,
#         "age_mae": (age_mae_sum / age_cnt) if age_cnt > 0 else None,
#         "gender_acc": (100.0 * gender_ok / gender_cnt) if gender_cnt > 0 else None,
#         "emo_acc": (100.0 * emo_ok / emo_cnt) if emo_cnt > 0 else None,
#         "counts": {"age": age_cnt, "gender": gender_cnt, "emotion": emo_cnt},
#     }


# Dynamic weight


def train_one_epoch_attr(model, loader, optimizer, device):
    model.train()

    age_loss_fn = nn.SmoothL1Loss(beta=1.0)
    gender_loss_fn = nn.CrossEntropyLoss()
    emo_loss_fn = nn.CrossEntropyLoss()

    total_loss_sum = 0.0
    total_samples = 0

    # metrics (masked)
    age_mae_sum, age_cnt = 0.0, 0
    gender_ok, gender_cnt = 0, 0
    emo_ok, emo_cnt = 0, 0

    for imgs, age, gender, emo in loader:
        imgs = imgs.to(device, non_blocking=True)
        age = age.to(device, non_blocking=True)
        gender = gender.to(device, non_blocking=True)
        emo = emo.to(device, non_blocking=True)

        out = model(imgs)
        log_vars = out["log_vars"]

        # --- LOSS SEPARATE CALCULATION ---

        # 1. Age Loss
        m_age = age != -1
        if m_age.any():
            # Only extract the lossy images that have age tags
            loss_age_raw = age_loss_fn(out["age"][m_age].squeeze(), age[m_age])
            # Apply dynamic weighting formula: Loss = (1/exp(log_var)) * raw_loss + log_var
            loss_age_weighted = torch.exp(-log_vars[0]) * loss_age_raw + log_vars[0]

            # Update metrics
            age_mae_sum += (out["age"][m_age].squeeze() - age[m_age]).abs().sum().item()
            age_cnt += int(m_age.sum().item())
        else:
            loss_age_weighted = torch.tensor(0.0, device=device)

        # 2. Gender Loss
        m_gender = gender != -1
        if m_gender.any():
            loss_gender_raw = gender_loss_fn(out["gender"][m_gender], gender[m_gender])
            loss_gender_weighted = (
                torch.exp(-log_vars[1]) * loss_gender_raw + log_vars[1]
            )

            # Update metrics
            gender_ok += (
                (out["gender"][m_gender].argmax(dim=1) == gender[m_gender]).sum().item()
            )
            gender_cnt += int(m_gender.sum().item())
        else:
            loss_gender_weighted = torch.tensor(0.0, device=device)

        # 3. Emotion Loss
        m_emo = emo != -1
        if m_emo.any():
            loss_emo_raw = emo_loss_fn(out["emotion"][m_emo], emo[m_emo])
            loss_emo_weighted = torch.exp(-log_vars[2]) * loss_emo_raw + log_vars[2]

            # Update metrics
            emo_ok += (out["emotion"][m_emo].argmax(dim=1) == emo[m_emo]).sum().item()
            emo_cnt += int(m_emo.sum().item())
        else:
            loss_emo_weighted = torch.tensor(0.0, device=device)

        # --- TOTAL LOSS ---
        # Only tasks that have batch data contribute to loss and gradient
        loss = loss_age_weighted + loss_gender_weighted + loss_emo_weighted

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        bs = imgs.size(0)
        total_loss_sum += float(loss.item()) * bs
        total_samples += bs

    return {
        "loss": total_loss_sum / max(total_samples, 1),
        "age_mae": (age_mae_sum / age_cnt) if age_cnt > 0 else None,
        "gender_acc": (100.0 * gender_ok / gender_cnt) if gender_cnt > 0 else None,
        "emo_acc": (100.0 * emo_ok / emo_cnt) if emo_cnt > 0 else None,
        "log_vars": log_vars.detach().cpu().numpy(),
    }


@torch.no_grad()
def eval_one_epoch_attr(model, dataloader, device, collect_predictions=False):
    model.eval()

    age_loss_fn = nn.SmoothL1Loss(beta=1.0)
    gender_loss_fn = nn.CrossEntropyLoss()
    emo_loss_fn = nn.CrossEntropyLoss()

    total_loss_sum = 0.0
    total_samples = 0

    age_mae_sum, age_cnt = 0.0, 0
    gender_ok, gender_cnt = 0, 0
    emo_ok, emo_cnt = 0, 0
    pred_store = {
        "age_true": [],
        "age_pred": [],
        "gender_true": [],
        "gender_pred": [],
        "emotion_true": [],
        "emotion_pred": [],
    }

    for inputs, age, gender, emo in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        age = age.to(device, non_blocking=True)
        gender = gender.to(device, non_blocking=True)
        emo = emo.to(device, non_blocking=True)

        out = model(inputs)
        log_vars = out["log_vars"]

        combined_loss = torch.tensor(0.0, device=device)

        m_age = age != -1
        if m_age.any():
            loss_age_raw = age_loss_fn(out["age"][m_age], age[m_age])
            combined_loss += torch.exp(-log_vars[0]) * loss_age_raw + log_vars[0]

            age_mae_sum += (out["age"][m_age] - age[m_age]).abs().sum().item()
            age_cnt += int(m_age.sum().item())
            if collect_predictions:
                pred_store["age_true"].append(age[m_age].detach().cpu().numpy())
                pred_store["age_pred"].append(out["age"][m_age].detach().cpu().numpy())

        m_gender = gender != -1
        if m_gender.any():
            loss_gender_raw = gender_loss_fn(out["gender"][m_gender], gender[m_gender])
            combined_loss += torch.exp(-log_vars[1]) * loss_gender_raw + log_vars[1]

            pred_g = out["gender"][m_gender].argmax(dim=1)
            gender_ok += (pred_g == gender[m_gender]).sum().item()
            gender_cnt += int(m_gender.sum().item())
            if collect_predictions:
                pred_store["gender_true"].append(
                    gender[m_gender].detach().cpu().numpy()
                )
                pred_store["gender_pred"].append(pred_g.detach().cpu().numpy())

        m_emo = emo != -1
        if m_emo.any():
            loss_emo_raw = emo_loss_fn(out["emotion"][m_emo], emo[m_emo])
            combined_loss += torch.exp(-log_vars[2]) * loss_emo_raw + log_vars[2]

            pred_e = out["emotion"][m_emo].argmax(dim=1)
            emo_ok += (pred_e == emo[m_emo]).sum().item()
            emo_cnt += int(m_emo.sum().item())
            if collect_predictions:
                pred_store["emotion_true"].append(emo[m_emo].detach().cpu().numpy())
                pred_store["emotion_pred"].append(pred_e.detach().cpu().numpy())

        bs = inputs.size(0)
        total_loss_sum += float(combined_loss.item()) * bs
        total_samples += bs

    metrics = {
        "loss": total_loss_sum / max(total_samples, 1),
        "age_mae": (age_mae_sum / age_cnt) if age_cnt > 0 else None,
        "gender_acc": (100.0 * gender_ok / gender_cnt) if gender_cnt > 0 else None,
        "emo_acc": (100.0 * emo_ok / emo_cnt) if emo_cnt > 0 else None,
        "counts": {"age": age_cnt, "gender": gender_cnt, "emotion": emo_cnt},
    }
    if collect_predictions:
        metrics["predictions"] = {
            key: np.concatenate(parts) if parts else np.array([])
            for key, parts in pred_store.items()
        }
    return metrics


# --------------------
# PART 4: MAIN FUNCTION
# --------------------


def main():
    # Create output directory
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    sys.stdout = Logger(os.path.join(Config.OUTPUT_DIR, "training.log"))

    # Initialize TensorBoard writer
    log_dir = os.path.join(
        Config.OUTPUT_DIR, "logs", datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    writer = SummaryWriter(log_dir)

    # Prepare data
    train_transform = get_train_transform()
    val_transform = get_val_transform()

    # Create datasets
    train_dataset = FaceAttrDataset(
        Config.DATA_ROOT,
        Config.TRAIN_LIST,
        transform=train_transform,
    )
    val_dataset = FaceAttrDataset(
        Config.DATA_ROOT,
        Config.VAL_LIST,
        transform=val_transform,
    )

    g = torch.Generator()
    g.manual_seed(42)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    # Initialize model
    if Config.USE_PRETRAINED:
        Pretrained_PTH = "resnet50_pretrained_on_msceleb.pth"
        model = FaceAttrModel(
            backbone_name="resnet50_pretrained",
            pretrained=Pretrained_PTH,
            emo_classes=7,
        ).to(Config.DEVICE)

        # Optimization - This is used for unpretrained training

        STAGE_1_EPOCHS = 10  # Freeze all backbone
        STAGE_2_EPOCHS = 15  # Unfreeze layer4
        STAGE_3_EPOCHS = 20  # Unfreeze layer3
        TOTAL_EPOCHS = STAGE_1_EPOCHS + STAGE_2_EPOCHS + STAGE_3_EPOCHS

        # Override Config.NUM_EPOCHS if needed
        Config.NUM_EPOCHS = TOTAL_EPOCHS

        # Optimizer & Scheduler
        param_groups = [
            {"params": model.age_head.parameters(), "lr": 3e-4, "name": "heads"},
            {"params": model.gender_head.parameters(), "lr": 3e-4, "name": "heads"},
            {"params": model.emo_head.parameters(), "lr": 3e-4, "name": "heads"},
            {"params": [model.log_vars], "lr": 1e-3},
            {
                "params": model.backbone.backbone.layer4.parameters(),
                "lr": 0,
                "name": "layer4",
            },  # Stage 1 keeps lr=0
        ]

        optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)

        # Use a single scheduler for the whole 30-epoch run
        # T_max = Config.NUM_EPOCHS so LR decays smoothly until the end
        scheduler = CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS, eta_min=1e-6)

        print(f"\n{'='*60}")
        print(f"TRAINING PLAN:")
        print(f"  Stage 1 (Epochs 1-{STAGE_1_EPOCHS}): Freeze all backbone")
        print(
            f"  Stage 2 (Epochs {STAGE_1_EPOCHS+1}-{STAGE_1_EPOCHS+STAGE_2_EPOCHS}): Unfreeze layer4 only"
        )
        print(
            f"  Stage 3 (Epochs {STAGE_1_EPOCHS+STAGE_2_EPOCHS+1}-{TOTAL_EPOCHS}): Unfreeze layer3 and layer4"
        )
        print(f"{'='*60}\n")
    else:
        model = FaceAttrModel(
            backbone_name=Config.BACKBONE, pretrained=None, emo_classes=7
        ).to(Config.DEVICE)

        optimizer = torch.optim.AdamW(
            [
                {"params": model.backbone.parameters(), "lr": 1e-5},
                {"params": model.bottleneck.parameters(), "lr": 1e-4},
                {"params": model.age_head.parameters(), "lr": 1e-4},
                {"params": model.gender_head.parameters(), "lr": 1e-4},
                {"params": model.emo_head.parameters(), "lr": 1e-4},
                {"params": [model.log_vars], "lr": 1e-3},
            ],
            weight_decay=1e-4,
            betas=(0.9, 0.999),
        )

        # Learning rate schedule: reduce lr after every 10 epochs
        scheduler = CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS, eta_min=1e-6)

    # Model information
    print(f"Model: {Config.BACKBONE} with embedding size {Config.EMBEDDING_SIZE}")
    print(f"Training on {Config.DEVICE}")

    # Early stopping setup
    early_stop = EarlyStopping(patience=5, min_delta=1e-4, mode="min", warmup_epochs=10)
    best_score = float("inf")
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_age_mae": [],
        "train_gender_acc": [],
        "train_emo_acc": [],
        "val_age_mae": [],
        "val_gender_acc": [],
        "val_emo_acc": [],
    }

    # Training loop
    for epoch in range(Config.NUM_EPOCHS):
        if Config.USE_PRETRAINED:
            # ========== STAGE 1: Freeze all backbone ==========
            if epoch == 0:
                print(f"\n{'='*60}")
                print(f"STAGE 1: Freezing entire backbone (Epochs 1-{STAGE_1_EPOCHS})")
                print(f"{'='*60}\n")

                freeze_layers(model, freeze_until="all")
                freeze_bn(model.backbone.backbone)

            # ========== STAGE 2: Unfreeze layer4 ==========
            elif epoch == STAGE_1_EPOCHS:
                print(f"\n{'='*60}")
                print(
                    f"STAGE 2: Unfreezing layer4 (Epochs {STAGE_1_EPOCHS+1}-{STAGE_1_EPOCHS+STAGE_2_EPOCHS})"
                )
                print(f"{'='*60}\n")

                # 1. Enable gradients for layer4
                for p in model.backbone.backbone.layer4.parameters():
                    p.requires_grad = True

                # 2. Update LR directly for the optimizer groups
                for pg in optimizer.param_groups:
                    if pg.get("name") == "layer4":
                        pg["lr"] = 1e-5  # Start backbone learning very slowly
                    elif pg.get("name") == "heads":
                        pg["lr"] = 1e-4

            # ========== STAGE 3: Unfreeze layer3 ==========
            elif epoch == STAGE_1_EPOCHS + STAGE_2_EPOCHS:
                print(f"\n{'='*60}")
                print(
                    f"STAGE 3: Unfreezing layer3 (Epochs {STAGE_1_EPOCHS+STAGE_2_EPOCHS+1}-{STAGE_1_EPOCHS+STAGE_2_EPOCHS+STAGE_3_EPOCHS})"
                )
                print(f"{'='*60}\n")

                freeze_layers(model, freeze_until="layer2")
                freeze_bn(model.backbone.backbone)
                layer3_block = model.backbone.backbone.layer3
                layer4_block = model.backbone.backbone.layer4

                optimizer = torch.optim.AdamW(
                    [
                        {"params": layer3_block.parameters(), "lr": 2e-5},
                        {"params": layer4_block.parameters(), "lr": 3e-5},
                        {"params": model.bottleneck.parameters(), "lr": 1e-4},
                        {"params": model.age_head.parameters(), "lr": 1e-4},
                        {"params": model.gender_head.parameters(), "lr": 1e-4},
                        {"params": model.emo_head.parameters(), "lr": 1e-4},
                        {"params": [model.log_vars], "lr": 1e-3},
                    ],
                    weight_decay=1e-4,
                    betas=(0.9, 0.999),
                )
                scheduler = CosineAnnealingLR(
                    optimizer, T_max=STAGE_3_EPOCHS, eta_min=1e-5
                )

        # ========== Train one epoch ==========
        train_metrics = train_one_epoch_attr(
            model,
            train_loader,
            optimizer,
            Config.DEVICE,
        )
        train_loss = train_metrics["loss"]

        val_metrics = eval_one_epoch_attr(model, val_loader, Config.DEVICE)
        val_loss = val_metrics["loss"]

        # Save metrics to history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_age_mae"].append(train_metrics["age_mae"])
        history["train_gender_acc"].append(train_metrics["gender_acc"])
        history["train_emo_acc"].append(train_metrics["emo_acc"])
        history["val_age_mae"].append(val_metrics["age_mae"])
        history["val_gender_acc"].append(val_metrics["gender_acc"])
        history["val_emo_acc"].append(val_metrics["emo_acc"])
        history["epoch"].append(epoch + 1)

        # Update learning rate
        scheduler.step()

        def fmt(x, nd=3, suffix=""):
            return "NA" if x is None else f"{x:.{nd}f}{suffix}"

        print(
            f"[Epoch {epoch+1}] Train loss={train_metrics['loss']:.4f} | "
            f"age_mae={fmt(train_metrics['age_mae'],3)} | "
            f"gender_acc={fmt(train_metrics['gender_acc'],2,'%')} | "
            f"emo_acc={fmt(train_metrics['emo_acc'],2,'%')}"
        )

        print(
            f"[Epoch {epoch+1}] Val loss={val_metrics['loss']:.4f} | "
            f"age_mae={fmt(val_metrics['age_mae'],3)} | "
            f"gender_acc={fmt(val_metrics['gender_acc'],2,'%')} | "
            f"emo_acc={fmt(val_metrics['emo_acc'],2,'%')}"
        )

        def tb_add(writer, tag, value, step):
            if value is not None:
                writer.add_scalar(tag, value, step)

        tb_add(writer, "Train/Age_MAE", train_metrics["age_mae"], epoch + 1)
        tb_add(writer, "Train/Gender_Acc", train_metrics["gender_acc"], epoch + 1)
        tb_add(writer, "Train/Emo_Acc", train_metrics["emo_acc"], epoch + 1)
        tb_add(writer, "Val/Age_MAE", val_metrics["age_mae"], epoch + 1)
        tb_add(writer, "Val/Gender_Acc", val_metrics["gender_acc"], epoch + 1)
        tb_add(writer, "Val/Emo_Acc", val_metrics["emo_acc"], epoch + 1)
        writer.add_scalar("Train/Loss", train_loss, epoch + 1)
        writer.add_scalar("Val/Loss", val_loss, epoch + 1)

        # Save checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": train_metrics["loss"],
                "train_age_mae": train_metrics["age_mae"],
                "train_gender_acc": train_metrics["gender_acc"],
                "train_emo_acc": train_metrics["emo_acc"],
                "val_loss": val_metrics["loss"],
                "val_age_mae": val_metrics["age_mae"],
                "val_gender_acc": val_metrics["gender_acc"],
                "val_emo_acc": val_metrics["emo_acc"],
            },
            os.path.join(Config.OUTPUT_DIR, "checkpoint.pth"),
        )
        print(f"[CHECKPOINT] Saved checkpoint.pth at epoch {epoch+1}")
        # Save best model based on val_loss
        score = val_metrics["loss"]
        if score < best_score:
            best_score = score
            torch.save(
                model.state_dict(), os.path.join(Config.OUTPUT_DIR, "best_model.pth")
            )
            print(f"[BEST MODEL] Saved best_model.pth at epoch {epoch+1}")
        # Early stopping check
        if early_stop.step(score):
            print(f"[EARLY STOPPING] Stopping training at epoch {epoch+1}")
            break

    # After training, plot training curves
    plot_loss_curve(history, os.path.join(Config.OUTPUT_DIR, "fig1_loss_curve.png"))
    plot_metric_curve(
        history,
        "val_emo_acc",
        os.path.join(Config.OUTPUT_DIR, "fig2_val_emo_acc.png"),
        title="Validation Curve - Emotion Accuracy",
        ylabel="Accuracy (%)",
        is_percent=True,
    )

    plot_metric_curve(
        history,
        "val_gender_acc",
        os.path.join(Config.OUTPUT_DIR, "fig3_val_gender_acc.png"),
        title="Validation Curve - Gender Accuracy",
        ylabel="Accuracy (%)",
        is_percent=True,
    )

    plot_metric_curve(
        history,
        "val_age_mae",
        os.path.join(Config.OUTPUT_DIR, "fig4_val_age_mae.png"),
        title="Validation Curve - Age MAE",
        ylabel="MAE (years)",
        is_percent=False,
    )
    plot_train_val_metric_curve(
        history,
        "train_age_mae",
        "val_age_mae",
        os.path.join(Config.OUTPUT_DIR, "fig5_train_val_age_mae.png"),
        title="Training vs Validation - Age MAE",
        ylabel="MAE (years)",
        is_percent=False,
    )
    plot_train_val_metric_curve(
        history,
        "train_gender_acc",
        "val_gender_acc",
        os.path.join(Config.OUTPUT_DIR, "fig6_train_val_gender_acc.png"),
        title="Training vs Validation - Gender Accuracy",
        ylabel="Accuracy (%)",
        is_percent=True,
    )
    plot_train_val_metric_curve(
        history,
        "train_emo_acc",
        "val_emo_acc",
        os.path.join(Config.OUTPUT_DIR, "fig7_train_val_emotion_acc.png"),
        title="Training vs Validation - Emotion Accuracy",
        ylabel="Accuracy (%)",
        is_percent=True,
    )

    print("[PLOT] Saved training curves fig1-fig7")

    # Close TensorBoard writer
    writer.close()

    print("Training completed!")

    # --- RUN AFTER THE TRAINING LOOP FINISHES ---
    print("\n" + "=" * 20 + " LEARNED OPTIMAL WEIGHTS " + "=" * 20)
    state_dict = torch.load(
        os.path.join(Config.OUTPUT_DIR, "best_model.pth"), weights_only=True
    )
    model.load_state_dict(state_dict)
    model.eval()
    final_val_metrics = eval_one_epoch_attr(
        model, val_loader, Config.DEVICE, collect_predictions=True
    )
    plot_final_evaluation(
        final_val_metrics,
        Config.OUTPUT_DIR,
        gender_names=GENDER_NAMES,
        emotion_names=EMOTION_NAMES,
    )

    with torch.no_grad():
        log_vars = model.log_vars.cpu().numpy()
        # Compute actual weights from log_vars
        raw_weights = np.exp(-log_vars)
        # Normalize to sum to 3, which makes comparison with the 1.0 default easier
        norm_w = raw_weights * (3.0 / np.sum(raw_weights))

        print(f"Age head weight    : {norm_w[0]:.4f}")
        print(f"Gender head weight : {norm_w[1]:.4f}")
        print(f"Emotion head weight: {norm_w[2]:.4f}")
    print("=" * 60)

    # Export model to ONNX
    best_pth = os.path.join(Config.OUTPUT_DIR, "best_model.pth")
    onnx_path = os.path.join(Config.OUTPUT_DIR, "embedder.onnx")

    export_faceattr_to_onnx(
        pth_path=best_pth,
        onnx_path=onnx_path,
        img_size=Config.IMG_SIZE,
        # backbone_name="resnet50_pretrained",
        backbone_name="resnet50",
        emo_classes=7,
        device="cpu",
    )
    print("[ONNX] exported once from final best_model.pth")


# --------------------
# PART 5: PLOTTING
# --------------------


def _apply_common_style(ax):
    ax.grid(True, linestyle=":", linewidth=1, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=11)


def _draw_stage_lines(ax, stage_boundaries=None, stage_labels=None):
    """
    stage_boundaries: list of 1-based epoch indices where training stages change.
    Example: [STAGE_1_EPOCHS, STAGE_1_EPOCHS + STAGE_2_EPOCHS]
    """
    if not stage_boundaries:
        return
    for i, b in enumerate(stage_boundaries):
        ax.axvline(b, linestyle="--", linewidth=1.5, alpha=0.6)
        if stage_labels and i < len(stage_labels):
            ax.text(
                b + 0.1,
                0.98,
                stage_labels[i],
                transform=ax.get_xaxis_transform(),
                va="top",
                fontsize=10,
                alpha=0.8,
            )


def plot_loss_curve(history, out_path, stage_boundaries=None, stage_labels=None):
    epoch = np.array(history["epoch"], dtype=int)
    train_loss = np.array(history["train_loss"], dtype=float)

    plt.figure(figsize=(11, 4.6))
    ax = plt.gca()
    _apply_common_style(ax)

    ax.plot(
        epoch,
        train_loss,
        linewidth=2.6,
        marker="o",
        markersize=4.2,
        label="Train Loss",
    )

    if "val_loss" in history and len(history["val_loss"]) == len(epoch):
        val_loss = np.array(history["val_loss"], dtype=float)
        ax.plot(
            epoch,
            val_loss,
            linewidth=2.6,
            linestyle="--",
            marker="s",
            markersize=4.2,
            label="Val Loss",
        )
        y_all = np.concatenate([train_loss, val_loss])
    else:
        y_all = train_loss

    ax.set_title("Training Curve - Loss", fontsize=15, pad=10)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # y-limits with light padding
    y_min, y_max = float(np.nanmin(y_all)), float(np.nanmax(y_all))
    pad = 0.08 * (y_max - y_min + 1e-12)
    ax.set_ylim(y_min - pad, y_max + pad)

    _draw_stage_lines(ax, stage_boundaries, stage_labels)

    ax.legend(loc="upper right", frameon=True, framealpha=0.85, fontsize=11)
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_metric_curve(history, key, out_path, title, ylabel, is_percent=False):
    epoch = np.array(history["epoch"], dtype=int)
    vals = history.get(key, [])

    # filter None values
    xs, ys = [], []
    for e, v in zip(epoch, vals):
        if v is None:
            continue
        xs.append(e)
        ys.append(float(v))

    if len(xs) == 0:
        print(f"[PLOT] Skip {key}: all values are None")
        return

    xs = np.array(xs, dtype=int)
    ys = np.array(ys, dtype=float)

    plt.figure(figsize=(11, 4.6))
    ax = plt.gca()
    _apply_common_style(ax)

    ax.plot(xs, ys, linewidth=2.6, marker="o", markersize=4.2, label=key)

    ax.set_title(title, fontsize=15, pad=10)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    if is_percent:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}%"))
        ax.set_ylim(max(0.0, ys.min() - 2), min(100.0, ys.max() + 2))

    ax.legend(loc="best", frameon=True, framealpha=0.85, fontsize=11)
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_train_val_metric_curve(
    history,
    train_key,
    val_key,
    out_path,
    title,
    ylabel,
    is_percent=False,
):
    epoch = np.array(history["epoch"], dtype=int)

    def _valid_xy(key):
        xs, ys = [], []
        for e, v in zip(epoch, history.get(key, [])):
            if v is not None:
                xs.append(e)
                ys.append(float(v))
        return np.array(xs, dtype=int), np.array(ys, dtype=float)

    train_x, train_y = _valid_xy(train_key)
    val_x, val_y = _valid_xy(val_key)
    if len(train_x) == 0 and len(val_x) == 0:
        print(f"[PLOT] Skip {train_key}/{val_key}: all values are None")
        return

    plt.figure(figsize=(11, 4.6))
    ax = plt.gca()
    _apply_common_style(ax)

    if len(train_x) > 0:
        ax.plot(
            train_x,
            train_y,
            linewidth=2.6,
            marker="o",
            markersize=4.2,
            label="Train",
        )
    if len(val_x) > 0:
        ax.plot(
            val_x,
            val_y,
            linewidth=2.6,
            linestyle="--",
            marker="s",
            markersize=4.2,
            label="Validation",
        )

    ax.set_title(title, fontsize=15, pad=10)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if is_percent:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}%"))
        values = np.concatenate([arr for arr in [train_y, val_y] if len(arr) > 0])
        ax.set_ylim(max(0.0, values.min() - 2), min(100.0, values.max() + 2))
    ax.legend(loc="best", frameon=True, framealpha=0.85, fontsize=11)
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def _confusion_matrix_np(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true.astype(int), y_pred.astype(int)):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def plot_confusion_matrix(cm, labels, out_path, title, normalize=False):
    display = cm.astype(float)
    if normalize:
        row_sum = display.sum(axis=1, keepdims=True)
        display = np.divide(
            display, row_sum, out=np.zeros_like(display), where=row_sum > 0
        )

    plt.figure(figsize=(8.2, 6.8))
    ax = plt.gca()
    im = ax.imshow(display, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=15, pad=10)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)

    thresh = display.max() / 2.0 if display.size > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = f"{display[i, j]:.2f}" if normalize else str(int(cm[i, j]))
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if display[i, j] > thresh else "black",
                fontsize=10,
            )

    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_age_scatter(y_true, y_pred, out_path):
    if len(y_true) == 0:
        print("[PLOT] Skip age scatter: no age labels")
        return
    plt.figure(figsize=(6.8, 6.2))
    ax = plt.gca()
    _apply_common_style(ax)
    ax.scatter(y_true, y_pred, s=12, alpha=0.35)
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    ax.plot(
        [lo, hi], [lo, hi], linestyle="--", linewidth=2, color="black", label="Ideal"
    )
    ax.set_title("Age Prediction - True vs Predicted", fontsize=15, pad=10)
    ax.set_xlabel("True Age", fontsize=12)
    ax.set_ylabel("Predicted Age", fontsize=12)
    ax.legend(loc="best", frameon=True, framealpha=0.85)
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_age_error_hist(y_true, y_pred, out_path):
    if len(y_true) == 0:
        print("[PLOT] Skip age error histogram: no age labels")
        return
    errors = y_pred - y_true
    abs_errors = np.abs(errors)
    plt.figure(figsize=(10, 4.6))
    ax = plt.gca()
    _apply_common_style(ax)
    ax.hist(abs_errors, bins=30, color="#1f77b4", alpha=0.85)
    ax.axvline(
        abs_errors.mean(), linestyle="--", color="black", linewidth=2, label="Mean"
    )
    ax.set_title("Age Absolute Error Distribution", fontsize=15, pad=10)
    ax.set_xlabel("Absolute Error (years)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.legend(loc="best", frameon=True, framealpha=0.85)
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_age_distribution(y_true, out_path):
    if len(y_true) == 0:
        print("[PLOT] Skip age distribution: no age labels")
        return
    plt.figure(figsize=(10, 4.6))
    ax = plt.gca()
    _apply_common_style(ax)
    ax.hist(y_true, bins=30, color="#1f77b4", alpha=0.85)
    ax.set_title("Age Distribution - Validation", fontsize=15, pad=10)
    ax.set_xlabel("Age", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_class_distribution(y_true, labels, out_path, title):
    if len(y_true) == 0:
        print(f"[PLOT] Skip {title}: no labels")
        return
    counts = np.bincount(y_true.astype(int), minlength=len(labels))[: len(labels)]
    plt.figure(figsize=(9, 4.8))
    ax = plt.gca()
    _apply_common_style(ax)
    ax.bar(np.arange(len(labels)), counts, color="#1f77b4", alpha=0.85)
    ax.set_title(title, fontsize=15, pad=10)
    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    for i, c in enumerate(counts):
        ax.text(i, c, str(int(c)), ha="center", va="bottom", fontsize=10)
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_per_class_accuracy(cm, labels, out_path, title):
    totals = cm.sum(axis=1)
    acc = np.divide(
        np.diag(cm), totals, out=np.zeros_like(totals, dtype=float), where=totals > 0
    )
    plt.figure(figsize=(9, 4.8))
    ax = plt.gca()
    _apply_common_style(ax)
    ax.bar(np.arange(len(labels)), acc * 100.0, color="#1f77b4", alpha=0.85)
    ax.set_title(title, fontsize=15, pad=10)
    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}%"))
    for i, value in enumerate(acc * 100.0):
        ax.text(i, value, f"{value:.1f}%", ha="center", va="bottom", fontsize=10)
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_final_evaluation(metrics, output_dir, gender_names, emotion_names):
    preds = metrics.get("predictions", {})

    age_true = preds.get("age_true", np.array([]))
    age_pred = preds.get("age_pred", np.array([]))
    plot_age_scatter(
        age_true,
        age_pred,
        os.path.join(output_dir, "fig8_age_true_vs_pred.png"),
    )
    plot_age_error_hist(
        age_true,
        age_pred,
        os.path.join(output_dir, "fig9_age_error_hist.png"),
    )
    plot_age_distribution(
        age_true,
        os.path.join(output_dir, "fig10_age_distribution.png"),
    )

    gender_true = preds.get("gender_true", np.array([]))
    gender_pred = preds.get("gender_pred", np.array([]))
    if len(gender_true) > 0:
        gender_cm = _confusion_matrix_np(gender_true, gender_pred, len(gender_names))
        plot_confusion_matrix(
            gender_cm,
            gender_names,
            os.path.join(output_dir, "fig11_gender_confusion_matrix.png"),
            "Gender Confusion Matrix - Validation",
        )
        plot_class_distribution(
            gender_true,
            gender_names,
            os.path.join(output_dir, "fig12_gender_distribution.png"),
            "Gender Distribution - Validation",
        )

    emotion_true = preds.get("emotion_true", np.array([]))
    emotion_pred = preds.get("emotion_pred", np.array([]))
    if len(emotion_true) > 0:
        emotion_cm = _confusion_matrix_np(
            emotion_true, emotion_pred, len(emotion_names)
        )
        plot_confusion_matrix(
            emotion_cm,
            emotion_names,
            os.path.join(output_dir, "fig13_emotion_confusion_matrix.png"),
            "Emotion Confusion Matrix - Validation",
        )
        plot_per_class_accuracy(
            emotion_cm,
            emotion_names,
            os.path.join(output_dir, "fig14_emotion_per_class_accuracy.png"),
            "Emotion Per-Class Accuracy - Validation",
        )
        plot_class_distribution(
            emotion_true,
            emotion_names,
            os.path.join(output_dir, "fig15_emotion_distribution.png"),
            "Emotion Distribution - Validation",
        )

    print("[PLOT] Saved final validation evaluation plots fig8-fig15")


# --------------------
# PART 6: EXPORT ONNX
# --------------------


class _FaceAttrONNXWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        # out["age"]: (B,), out["gender"]: (B,2), out["emotion"]: (B,K)
        # ONNX handles 2D scalar outputs better, so reshape age to (B,1)
        age = out["age"].unsqueeze(1)
        gender_logits = out["gender"]
        emotion_logits = out["emotion"]
        return age, gender_logits, emotion_logits


def export_faceattr_to_onnx(
    pth_path: str,
    onnx_path: str,
    img_size: int = Config.IMG_SIZE,
    backbone_name: str = "resnet50",
    emo_classes: int = 7,
    device: str = "cpu",
    strict: bool = True,
):
    """
    Export FaceAttrModel to ONNX.
    Outputs:
      - age: (B,1)
      - gender_logits: (B,2)
      - emotion_logits: (B,K)
    """

    # 1) Build the model using the matching FaceAttrModel constructor
    model = FaceAttrModel(
        backbone_name=backbone_name,
        pretrained=None,  # export does not need to load pretrained weights through this path
        emo_classes=emo_classes,
    ).to(device)

    # 2) Load checkpoint
    ckpt = torch.load(pth_path, map_location=device)

    # If the checkpoint was saved as {"model_state_dict": ...}
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    else:
        sd = ckpt

    # 3) Load state dict. Keep this strict by default so a wrong checkpoint
    # cannot silently export a partially random model.
    missing, unexpected = model.load_state_dict(sd, strict=strict)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch. Missing keys: {missing}. Unexpected keys: {unexpected}."
        )

    model.eval()

    # 4) Wrap for ONNX (tuple outputs)
    wrapper = _FaceAttrONNXWrapper(model).to(device).eval()

    # 5) Export
    dummy = torch.randn(1, 3, img_size, img_size, device=device)

    torch.onnx.export(
        wrapper,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["age", "gender_logits", "emotion_logits"],
        opset_version=18,
        dynamic_axes={
            "input": {0: "batch"},
            "age": {0: "batch"},
            "gender_logits": {0: "batch"},
            "emotion_logits": {0: "batch"},
        },
    )

    print(f"Exported FaceAttrModel ONNX to: {onnx_path}")


# --------------------
# PART 7: INFERENCE
# --------------------


GENDER_LABELS = {0: "male", 1: "female"}

EMO_LABELS = {
    0: "anger",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "sad",
    6: "surprise",
}


class FaceAttrPredictor:
    def __init__(
        self,
        ckpt_path: str,
        device=None,
        backbone_name="resnet50",
        emo_classes=7,
        img_size=Config.IMG_SIZE,
        pretrained_backbone_path=None,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Build model; backbone_name and emo_classes must match training
        self.model = FaceAttrModel(
            backbone_name=backbone_name,
            pretrained=pretrained_backbone_path,
            emo_classes=emo_classes,
        ).to(self.device)

        ckpt = torch.load(ckpt_path, map_location=self.device)
        sd = (
            ckpt["model_state_dict"]
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt
            else ckpt
        )

        # Strip DataParallel prefix
        if any(k.startswith("module.") for k in sd.keys()):
            sd = {k[len("module.") :]: v for k, v in sd.items()}

        self.model.load_state_dict(sd, strict=True)
        self.model.eval()

        # Transform as validation
        self.transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.Lambda(lambda img: img.convert("RGB")),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def _to_pil(self, image):
        if isinstance(image, str):
            return Image.open(image).convert("RGB")
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = image.astype(np.uint8)
            return Image.fromarray(image).convert("RGB")
        raise TypeError(f"Unsupported image type: {type(image)}")

    @torch.no_grad()
    def predict(self, image):
        """
        image: path | PIL.Image | numpy(H,W,C)
        return dict with age, gender, emotion
        """
        pil = self._to_pil(image)
        x = self.transform(pil).unsqueeze(0).to(self.device)  # (1,3,H,W)

        out = self.model(x)

        # age regression
        age = float(out["age"].item())
        age_int = int(max(1, min(100, round(age))))

        # gender
        gender_logits = out["gender"][0]  # (2,)
        gender_probs = F.softmax(gender_logits, dim=0)
        gender_id = int(torch.argmax(gender_probs).item())
        gender_prob = float(gender_probs[gender_id].item())
        gender_label = GENDER_LABELS.get(gender_id, str(gender_id))

        # emotion
        emo_logits = out["emotion"][0]  # (K,)
        emo_probs = F.softmax(emo_logits, dim=0)
        emo_id = int(torch.argmax(emo_probs).item())
        emo_prob = float(emo_probs[emo_id].item())
        emo_label = EMO_LABELS.get(emo_id, str(emo_id))

        return {
            "age": age_int,
            "gender_id": gender_id,
            "gender_label": gender_label,
            "gender_prob": gender_prob,
            "emotion_id": emo_id,
            "emotion_label": emo_label,
            "emotion_prob": emo_prob,
            "emotion_probs": emo_probs.detach().cpu().tolist(),
        }


# Run main function when running script directly
if __name__ == "__main__":
    main()
