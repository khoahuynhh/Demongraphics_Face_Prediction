import torch
from model_resnet import FaceAttrModel, export_faceattr_to_onnx, Config

PTH_PATH = "outputs/resnet50_2/best_model.pth"
ONNX_OUTPUT_PATH = "outputs/resnet50_2/best_model.onnx"

print(f"--- Đang bắt đầu chuyển đổi {PTH_PATH} sang ONNX ---")

try:
    export_faceattr_to_onnx(
        pth_path=PTH_PATH,
        onnx_path=ONNX_OUTPUT_PATH,
        img_size=Config.IMG_SIZE,
        backbone_name="resnet50",
        emo_classes=7,
        device="cpu",
    )
    print(f"--- Chuyển đổi thành công! File lưu tại: {ONNX_OUTPUT_PATH} ---")
except Exception as e:
    print(f"Lỗi khi chuyển đổi: {e}")
