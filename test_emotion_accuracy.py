import argparse
import csv
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class _DummySummaryWriter:
    def __init__(self, *args, **kwargs):
        pass

    def add_scalar(self, *args, **kwargs):
        pass

    def close(self):
        pass


_tensorboard_stub = types.ModuleType("torch.utils.tensorboard")
_tensorboard_stub.SummaryWriter = _DummySummaryWriter
sys.modules.setdefault("torch.utils.tensorboard", _tensorboard_stub)

import model_resnet
import model_yolo


EMOTION_NAMES = ["anger", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
FOLDER_TO_LABEL = {
    "anger": 0,
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "neutral": 4,
    "sad": 5,
    "surprise": 6,
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class FER2013FolderDataset(Dataset):
    def __init__(self, root, transform, max_samples=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []

        if not self.root.is_dir():
            raise FileNotFoundError(f"Test folder not found: {self.root}")

        for folder_name, label in FOLDER_TO_LABEL.items():
            folder = self.root / folder_name
            if not folder.is_dir():
                continue
            for image_path in sorted(folder.iterdir()):
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
                    self.samples.append((image_path, label))

        if not self.samples:
            raise RuntimeError(f"No images found under class folders in: {self.root}")
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image.contiguous().clone(), torch.tensor(label, dtype=torch.long), str(image_path)


def pick_first_existing(paths):
    for path in paths:
        path = Path(path)
        if path.is_file():
            return path
    return Path(paths[0])


def load_state_dict(model, checkpoint_path, device, strict=True):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    model.load_state_dict(state_dict, strict=strict)
    return model


def build_resnet(checkpoint_path, device, backbone):
    # resnet50_pretrained uses weights=None in this codebase, avoiding any
    # torchvision download. The trained checkpoint then supplies all weights.
    model = model_resnet.FaceAttrModel(backbone_name=backbone, pretrained=None, emo_classes=7)
    model = load_state_dict(model, checkpoint_path, device)
    return model.to(device).eval()


def build_yolo(checkpoint_path, device, backbone):
    model = model_yolo.FaceAttrModel(backbone_name=backbone, pretrained=True, emo_classes=7)
    model = load_state_dict(model, checkpoint_path, device)
    return model.to(device).eval()


@torch.no_grad()
def evaluate_emotion(model, dataloader, device, save_predictions=None):
    confusion = np.zeros((len(EMOTION_NAMES), len(EMOTION_NAMES)), dtype=np.int64)
    total = 0
    correct = 0

    pred_file = None
    writer = None
    if save_predictions is not None:
        pred_file = Path(save_predictions).open("w", encoding="utf-8", newline="")
        writer = csv.writer(pred_file)
        writer.writerow(["path", "true_id", "true_label", "pred_id", "pred_label", "confidence"])

    try:
        for images, labels, paths in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            logits = outputs["emotion"]
            probs = torch.softmax(logits, dim=1)
            confidence, preds = probs.max(dim=1)

            total += labels.numel()
            correct += (preds == labels).sum().item()

            true_np = labels.cpu().numpy()
            pred_np = preds.cpu().numpy()
            conf_np = confidence.cpu().numpy()
            for true_id, pred_id in zip(true_np, pred_np):
                confusion[int(true_id), int(pred_id)] += 1

            if writer is not None:
                for path, true_id, pred_id, conf in zip(paths, true_np, pred_np, conf_np):
                    writer.writerow(
                        [
                            path,
                            int(true_id),
                            EMOTION_NAMES[int(true_id)],
                            int(pred_id),
                            EMOTION_NAMES[int(pred_id)],
                            f"{float(conf):.6f}",
                        ]
                    )
    finally:
        if pred_file is not None:
            pred_file.close()

    per_class = {}
    for idx, name in enumerate(EMOTION_NAMES):
        support = int(confusion[idx].sum())
        hits = int(confusion[idx, idx])
        per_class[name] = {
            "correct": hits,
            "total": support,
            "accuracy": (100.0 * hits / support) if support else None,
        }

    return {
        "total": total,
        "correct": correct,
        "accuracy": 100.0 * correct / max(total, 1),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def print_metrics(name, metrics):
    print(f"\n{name}")
    print("-" * len(name))
    print(f"Total: {metrics['total']}")
    print(f"Correct: {metrics['correct']}")
    print(f"Emotion accuracy: {metrics['accuracy']:.2f}%")
    print("\nPer-class accuracy:")
    for label in EMOTION_NAMES:
        item = metrics["per_class"][label]
        acc = "n/a" if item["accuracy"] is None else f"{item['accuracy']:.2f}%"
        print(f"  {label:8s}: {acc:>7s} ({item['correct']}/{item['total']})")
    print("\nConfusion matrix rows=true, cols=pred:")
    print(" " * 10 + " ".join(f"{name[:4]:>6s}" for name in EMOTION_NAMES))
    for label, row in zip(EMOTION_NAMES, metrics["confusion_matrix"]):
        print(f"{label[:8]:>8s}  " + " ".join(f"{value:6d}" for value in row))


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate one ResNet or YOLO emotion checkpoint on FER2013 folder data."
    )
    parser.add_argument("--test-dir", default="data/test_2", help="FER2013 test folder.")
    parser.add_argument(
        "--model-type",
        choices=["resnet", "yolo"],
        required=True,
        help="Architecture family of the checkpoint to evaluate.",
    )
    parser.add_argument(
        "--ckpt",
        required=True,
        help="Path to one .pth checkpoint.",
    )
    parser.add_argument(
        "--resnet-ckpt",
        default="outputs/resnet50/best_model_v2.pth",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--yolo-ckpt",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--resnet-backbone", default="resnet50_pretrained")
    parser.add_argument("--yolo-backbone", default="yolo26n_cls")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional smoke-test limit. Default evaluates all images.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--only", choices=["both", "resnet", "yolo"], default=None, help=argparse.SUPPRESS)
    parser.add_argument("--output-json", default="outputs/test_2_emotion_accuracy.json")
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save per-image predictions as CSV files next to the JSON output.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint_path = Path(args.ckpt)

    dataset = FER2013FolderDataset(
        args.test_dir, model_resnet.get_val_transform(), max_samples=args.max_samples
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    print(f"Device: {device}")
    print(f"Test folder: {Path(args.test_dir).resolve()}")
    print(f"Images: {len(dataset)}")

    results = None
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    if args.model_type == "resnet":
        print(f"\nLoading ResNet checkpoint: {checkpoint_path}")
        model = build_resnet(checkpoint_path, device, args.resnet_backbone)
        pred_csv = output_json.with_name("test_2_resnet_predictions.csv") if args.save_predictions else None
        results = evaluate_emotion(model, dataloader, device, pred_csv)
        print_metrics("ResNet", results)
    else:
        print(f"\nLoading YOLO checkpoint: {checkpoint_path}")
        model = build_yolo(checkpoint_path, device, args.yolo_backbone)
        pred_csv = output_json.with_name("test_2_yolo_predictions.csv") if args.save_predictions else None
        results = evaluate_emotion(model, dataloader, device, pred_csv)
        print_metrics("YOLO", results)

    output = {
        "test_dir": str(Path(args.test_dir).resolve()),
        "num_images": len(dataset),
        "labels": EMOTION_NAMES,
        "model_type": args.model_type,
        "checkpoint": str(checkpoint_path),
        "results": results,
    }
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved metrics: {output_json}")


if __name__ == "__main__":
    main()
