import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

try:
    from mtcnn_ort import MTCNN as MTCNNDetector
except ImportError:
    try:
        from mtcnn import MTCNN as MTCNNDetector
    except ImportError:
        MTCNNDetector = None


DEFAULT_MODEL_CANDIDATES = (
    Path("outputs/resnet50_pretrained/best_model_v2_embedder.onnx"),
)

GENDER_LABELS = {
    0: "male",
    1: "female",
}

EMOTION_LABELS = {
    0: "anger",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "sad",
    6: "surprise",
}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float32)
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def pick_default_model() -> Path:
    for path in DEFAULT_MODEL_CANDIDATES:
        if path.exists():
            return path
    candidates = ", ".join(str(path) for path in DEFAULT_MODEL_CANDIDATES)
    raise FileNotFoundError(f"No default ONNX model found. Tried: {candidates}")


def infer_preprocess_mode(model_path: Path) -> str:
    return "none" if "yolo" in str(model_path).lower() else "imagenet"


class EdgeMTCNNDetector:
    def __init__(self, min_face_size: int = 60, margin: int = 15):
        if MTCNNDetector is None:
            raise RuntimeError(
                "MTCNN detector is unavailable. Install mtcnn-onnxruntime for edge "
                "deployment: pip install mtcnn-onnxruntime"
            )

        self.detector = MTCNNDetector()
        self.min_face_size = int(min_face_size)
        self.margin = int(margin)

    def detect_faces(self, frame: np.ndarray) -> list[dict]:
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = self.detector.detect_faces(img_rgb)

        faces = []
        frame_h, frame_w = frame.shape[:2]
        for det in detections:
            x, y, w, h = det["box"]
            if w < self.min_face_size or h < self.min_face_size:
                continue

            x1 = max(0, int(x) - self.margin)
            y1 = max(0, int(y) - self.margin)
            x2 = min(frame_w, int(x + w) + self.margin)
            y2 = min(frame_h, int(y + h) + self.margin)
            faces.append(
                {
                    "box": [x1, y1, x2 - x1, y2 - y1],
                    "confidence": float(det.get("confidence", 1.0)),
                    "keypoints": det.get("keypoints", {}),
                }
            )

        return faces


class FaceAttrEdgePredictor:
    """
    ONNX edge predictor for FaceAttrModel exports.

    Expected ONNX outputs:
      - age: (B, 1)
      - gender_logits: (B, 2)
      - emotion_logits: (B, 7)
    """

    def __init__(
        self,
        model_path: str | Path,
        providers: list[str] | None = None,
        img_size: int | None = None,
        min_face_size: int = 60,
        detector_margin: int = 15,
        preprocess_mode: str = "auto",
    ):
        self.model_path = Path(model_path)
        if self.model_path.suffix.lower() != ".onnx":
            raise ValueError(
                f"Edge inference expects an ONNX model, got: {self.model_path}"
            )
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self.providers = providers or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=self.providers,
        )

        inputs = self.session.get_inputs()
        if len(inputs) != 1:
            raise ValueError(f"Expected one ONNX input, got {len(inputs)} inputs")

        self.input_name = inputs[0].name
        self.img_size = img_size or self._infer_img_size(inputs[0].shape) or 224
        self.preprocess_mode = (
            infer_preprocess_mode(self.model_path)
            if preprocess_mode == "auto"
            else preprocess_mode
        )
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.min_face_size = int(min_face_size)
        self.face_detector = EdgeMTCNNDetector(
            min_face_size=self.min_face_size,
            margin=detector_margin,
        )

    @staticmethod
    def _infer_img_size(input_shape) -> int | None:
        if len(input_shape) != 4:
            return None
        height, width = input_shape[2], input_shape[3]
        if isinstance(height, int) and isinstance(width, int) and height == width:
            return height
        return None

    def preprocess_face(self, face_bgr: np.ndarray) -> np.ndarray:
        img = cv2.resize(face_bgr, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        if self.preprocess_mode == "imagenet":
            img = (img - IMAGENET_MEAN) / IMAGENET_STD
        elif self.preprocess_mode != "none":
            raise ValueError(
                f"Unsupported preprocess mode: {self.preprocess_mode}. "
                "Use auto, imagenet, or none."
            )
        img = np.transpose(img, (2, 0, 1))
        return np.expand_dims(img, axis=0).astype(np.float32)

    def _output_by_name(self, outputs: list[np.ndarray]) -> dict[str, np.ndarray]:
        named = dict(zip(self.output_names, outputs))
        if {"age", "gender_logits", "emotion_logits"}.issubset(named):
            return {
                "age": named["age"],
                "gender_logits": named["gender_logits"],
                "emotion_logits": named["emotion_logits"],
            }

        # Fallback for exports with unnamed or reordered metadata but same shape order.
        if len(outputs) < 3:
            raise ValueError(
                f"Expected at least 3 ONNX outputs, got {len(outputs)}: "
                f"{self.output_names}"
            )
        return {
            "age": outputs[0],
            "gender_logits": outputs[1],
            "emotion_logits": outputs[2],
        }

    def _parse_outputs(self, outputs: list[np.ndarray]) -> dict:
        out = self._output_by_name(outputs)

        age_raw = float(np.asarray(out["age"]).reshape(-1)[0])
        age = int(max(0, min(100, round(age_raw))))

        gender_logits = np.asarray(out["gender_logits"])[0]
        gender_probs = softmax(gender_logits)
        gender_id = int(np.argmax(gender_probs))

        emotion_logits = np.asarray(out["emotion_logits"])[0]
        emotion_probs = softmax(emotion_logits)
        emotion_id = int(np.argmax(emotion_probs))

        return {
            "age": age,
            "gender_id": gender_id,
            "gender": GENDER_LABELS.get(gender_id, str(gender_id)),
            "gender_prob": round(float(gender_probs[gender_id]), 4),
            "emotion_id": emotion_id,
            "emotion": EMOTION_LABELS.get(emotion_id, str(emotion_id)),
            "emotion_prob": round(float(emotion_probs[emotion_id]), 4),
        }

    def detect_faces(self, frame: np.ndarray) -> list[dict]:
        return self.face_detector.detect_faces(frame)

    def predict(self, frame: np.ndarray) -> list[dict]:
        faces = self.detect_faces(frame)
        results = []

        for face_det in faces:
            x, y, w, h = face_det["box"]
            start = time.perf_counter()
            face = frame[y : y + h, x : x + w]
            if face.size == 0:
                continue

            input_tensor = self.preprocess_face(face)
            outputs = self.session.run(None, {self.input_name: input_tensor})
            latency_ms = (time.perf_counter() - start) * 1000.0

            parsed = self._parse_outputs(outputs)
            parsed.update(
                {
                    "box": [int(x), int(y), int(w), int(h)],
                    "face_confidence": round(float(face_det["confidence"]), 4),
                    "latency_ms": round(latency_ms, 2),
                    "model": str(self.model_path),
                }
            )
            results.append(parsed)

        return results


def draw_results(frame: np.ndarray, predictions: list[dict]) -> np.ndarray:
    img = frame.copy()

    for pred in predictions:
        x, y, w, h = pred["box"]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        lines = [
            f"{pred['gender']} ({pred['gender_prob']:.2f})",
            f"Age: {pred['age']}",
            f"{pred['emotion']} ({pred['emotion_prob']:.2f})",
            f"{pred['latency_ms']:.1f} ms",
        ]

        label_height = 20 * len(lines) + 8
        top = max(0, y - label_height)
        cv2.rectangle(img, (x, top), (x + max(w, 150), y), (0, 0, 0), -1)

        for idx, line in enumerate(lines):
            color = (0, 255, 255) if idx == 2 else (255, 255, 255)
            cv2.putText(
                img,
                line,
                (x + 5, top + 18 + idx * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

    return img


def profile_payload(predictions: list[dict]) -> dict | None:
    if not predictions:
        return None
    first = predictions[0]
    return {
        "gender": first["gender"],
        "age": first["age"],
        "emotion": first["emotion"],
    }


def run_image(args, predictor: FaceAttrEdgePredictor) -> None:
    image_path = Path(args.image)
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    predictions = predictor.predict(frame)
    print(json.dumps(predictions, indent=2, ensure_ascii=False))

    if args.output:
        result = draw_results(frame, predictions)
        cv2.imwrite(str(args.output), result)

    if args.show:
        cv2.imshow("Face Attribute Result", draw_results(frame, predictions))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_camera(args, predictor: FaceAttrEdgePredictor) -> None:
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index: {args.camera}")

    print("Face attribute edge inference")
    print(f"Model: {predictor.model_path}")
    print("Press SPACE to capture and analyze. Press q to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        preview = frame.copy()
        cv2.putText(
            preview,
            "SPACE: capture | q: quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("Camera Preview", preview)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key != 32:
            continue

        predictions = predictor.predict(frame)
        payload = profile_payload(predictions)
        if payload is None:
            print("No face detected.")
        else:
            print(json.dumps(payload, indent=2, ensure_ascii=False))

        result = draw_results(frame, predictions)
        cv2.imshow("Captured Result - press any key", result)
        cv2.waitKey(0)
        cv2.destroyWindow("Captured Result - press any key")

    cap.release()
    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run face age/gender/emotion inference with an ONNX model."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to ONNX model. Defaults to outputs/resnet50/embedder.onnx if present.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Run once on an image instead of camera.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output image path for --image mode.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show result window in --image mode.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index for live capture mode.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=None,
        help="Override model input image size. Usually 224.",
    )
    parser.add_argument(
        "--provider",
        action="append",
        default=None,
        help="ONNX Runtime provider. Can be passed multiple times.",
    )
    parser.add_argument(
        "--min-face-size",
        type=int,
        default=60,
        help="Minimum detected face size in pixels.",
    )
    parser.add_argument(
        "--detector-margin",
        type=int,
        default=15,
        help="Pixel margin added around each MTCNN face crop.",
    )
    parser.add_argument(
        "--preprocess",
        choices=["auto", "imagenet", "none"],
        default="auto",
        help="Input normalization. auto uses none for YOLO paths and imagenet otherwise.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model or pick_default_model()
    providers = args.provider or ["CPUExecutionProvider"]

    predictor = FaceAttrEdgePredictor(
        model_path=model_path,
        providers=providers,
        img_size=args.img_size,
        min_face_size=args.min_face_size,
        detector_margin=args.detector_margin,
        preprocess_mode=args.preprocess,
    )

    if args.image:
        run_image(args, predictor)
    else:
        run_camera(args, predictor)


if __name__ == "__main__":
    main()
