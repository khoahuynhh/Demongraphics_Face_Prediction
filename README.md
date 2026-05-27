# Face Attributes Prediction

Project dự đoán thuộc tính khuôn mặt từ ảnh: tuổi, giới tính và cảm xúc. Repository này chứa code huấn luyện, đánh giá, export ONNX và suy luận edge cho hai hướng backbone chính: ResNet và YOLO classification.

## Chức năng chính

- Huấn luyện mô hình đa nhiệm dự đoán `age`, `gender`, `emotion`.
- Hỗ trợ backbone ResNet trong `model_resnet.py`.
- Hỗ trợ backbone YOLO classification trong `model_yolo.py`.
- Export checkpoint `.pth` sang ONNX.
- Suy luận ảnh/camera bằng ONNX Runtime và MTCNN trong `inference_edge_2.py`.
- Sinh biểu đồ đánh giá, log huấn luyện và benchmark latency.

## Cấu trúc thư mục

```text
.
├── data/
│   ├── face_preprocessor.py
│   ├── create_train_list.py
│   ├── create_emo_val_list.py
│   ├── split_train_val.py
│   └── merge_val.py
├── outputs/
├── model_resnet.py
├── model_yolo.py
├── inference_edge_2.py
├── pth_to_onnx_resnet.py
├── pth_to_onnx_yolo.py
├── requirements.txt
└── README.md
```

Lưu ý: `data/`, `outputs/`, model weights và ONNX files đã được ignore trong `.gitignore` vì thường có dung lượng lớn hoặc là dữ liệu local.

## Cài đặt môi trường

Khuyến nghị dùng Python 3.10 hoặc 3.11.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Nếu máy không dùng CUDA 12.1, chỉnh lại phần `torch` và `torchvision` trong `requirements.txt` theo wheel phù hợp từ PyTorch.

## Chuẩn bị dữ liệu

Các file list train/validation có định dạng:

```text
relative_path age gender emotion
```

Ví dụ:

```text
faces/person_001.jpg 24 0 3
```

Trong đó:

- `age`: tuổi, dạng số.
- `gender`: `0` là male, `1` là female.
- `emotion`: id cảm xúc theo danh sách trong code.
- `-1`: dùng khi thiếu nhãn để bỏ qua loss/metric tương ứng.

Mặc định code đọc:

```text
data/train_new.txt
data/val.txt
```

Có thể chỉnh trong class `Config` của `model_resnet.py` hoặc `model_yolo.py`.

## Huấn luyện

Huấn luyện ResNet:

```bash
python model_resnet.py
```

Huấn luyện YOLO classification:

```bash
python model_yolo.py
```

Kết quả huấn luyện được ghi vào `outputs/`, bao gồm checkpoint, log, TensorBoard event files và các biểu đồ đánh giá.

## Export ONNX

Export ResNet:

```bash
python pth_to_onnx_resnet.py
```

Export YOLO:

```bash
python pth_to_onnx_yolo.py --pth outputs/yolov26n_cls/best_model.pth --onnx outputs/yolov26n_cls/best_model.onnx
```

Có thể xem thêm tham số:

```bash
python pth_to_onnx_yolo.py --help
```

## Suy luận

Suy luận trên ảnh:

```bash
python inference_edge_2.py --model outputs/yolov26n_cls/best_model.onnx --image path/to/image.jpg --output outputs/result.jpg
```

Suy luận bằng camera:

```bash
python inference_edge_2.py --model outputs/yolov26n_cls/best_model.onnx --camera 0
```

Script sẽ detect khuôn mặt, crop vùng mặt, chạy ONNX model và trả về tuổi, giới tính, cảm xúc kèm latency.

## Ghi chú

- `requirements.txt` hiện đang pin PyTorch CUDA 12.1. Nếu cài trên CPU-only, cần đổi sang bản CPU.
- `resnet50_pretrained_on_msceleb.pth`, `yolo26n-cls.pt`, `yolov8n-cls.pt` là model weights local và đã được `.gitignore`.
- Trước khi push, nên chạy `git status --ignored` để kiểm tra file nào đang được track và file nào bị ignore.
