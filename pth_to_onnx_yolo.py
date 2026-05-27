import argparse
from pathlib import Path
import sys
import types

from model_yolo import export_faceattr_to_onnx

DEFAULT_CANDIDATES = (
    Path("outputs/yolov26n_cls/best_model.pth"),
    Path("outputs/yolo26n_cls/best_model_yolo.pth"),
    Path("outputs/yolov26n_cls/best_model.pth"),
    Path("outputs/yolo26n_cls/best_model.pth"),
)
DEFAULT_IMG_SIZE = 224
DEFAULT_BACKBONE = "yolo26n_cls"


def find_default_pth() -> Path:
    for path in DEFAULT_CANDIDATES:
        if path.exists():
            return path
    return DEFAULT_CANDIDATES[0]


def parse_args():
    default_pth = find_default_pth()
    default_onnx = default_pth.with_suffix(".onnx")

    parser = argparse.ArgumentParser(
        description="Convert YOLO face attribute .pth checkpoint to ONNX."
    )
    parser.add_argument(
        "--pth",
        default=str(default_pth),
        help="Path to YOLO .pth checkpoint.",
    )
    parser.add_argument(
        "--onnx",
        default=str(default_onnx),
        help="Output ONNX path.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=DEFAULT_IMG_SIZE,
        help="Input image size used during training.",
    )
    parser.add_argument(
        "--backbone",
        default=DEFAULT_BACKBONE,
        help="YOLO classification backbone name, for example yolo26n_cls.",
    )
    parser.add_argument(
        "--emo-classes",
        type=int,
        default=7,
        help="Number of emotion classes.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda"),
        help="Device used for export. CPU is recommended for stable export.",
    )
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="Skip ONNX model validation after export.",
    )
    return parser.parse_args()


def import_yolo_exporter():
    # model_yolo imports SummaryWriter for training, but ONNX export does not use it.
    # Some local TensorBoard/TensorFlow installs can fail at import time, so provide
    # a minimal stub for this conversion-only script.
    tensorboard_stub = types.ModuleType("torch.utils.tensorboard")

    class SummaryWriter:
        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def close(self):
            pass

    tensorboard_stub.SummaryWriter = SummaryWriter
    sys.modules.setdefault("torch.utils.tensorboard", tensorboard_stub)

    return export_faceattr_to_onnx


def validate_onnx(onnx_path: Path):
    import onnx

    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)


def ensure_onnx_installed():
    try:
        import onnx  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency: onnx. Install project requirements first, "
            "for example: pip install -r requirements.txt"
        ) from exc


def main():
    args = parse_args()
    import torch

    pth_path = Path(args.pth)
    onnx_path = Path(args.onnx)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    if not pth_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {pth_path}")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")

    print(f"Converting YOLO checkpoint: {pth_path}")
    print(f"Output ONNX file        : {onnx_path}")
    print(f"Backbone                : {args.backbone}")
    print(f"Image size              : {args.img_size}")

    ensure_onnx_installed()

    export_faceattr_to_onnx = import_yolo_exporter()
    export_faceattr_to_onnx(
        pth_path=str(pth_path),
        onnx_path=str(onnx_path),
        img_size=args.img_size,
        backbone_name=args.backbone,
        emo_classes=args.emo_classes,
        device=args.device,
        strict=True,
    )

    if not args.no_check:
        validate_onnx(onnx_path)
        print("ONNX validation passed.")

    print(f"Done: {onnx_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"Error: {exc}") from exc
