import argparse
from pathlib import Path

import cv2
import numpy as np

MTCNN_IMPORT_ERROR = None

try:
    from mtcnn import MTCNN
except ImportError as exc:
    MTCNN = None
    MTCNN_IMPORT_ERROR = exc


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class FacePreprocessor:
    def __init__(self, target_size=(224, 224), margin=20):
        self.target_size = target_size
        self.margin = margin
        self.detector = MTCNN() if MTCNN is not None else None
        self.face_cascade = None
        self.eye_cascade = None

        if self.detector is None:
            haar_dir = Path(cv2.data.haarcascades)
            self.face_cascade = cv2.CascadeClassifier(
                str(haar_dir / "haarcascade_frontalface_default.xml")
            )
            self.eye_cascade = cv2.CascadeClassifier(
                str(haar_dir / "haarcascade_eye.xml")
            )
            if MTCNN_IMPORT_ERROR is not None:
                print(
                    "Failed to import MTCNN "
                    f"({MTCNN_IMPORT_ERROR}). Falling back to OpenCV Haar Cascade."
                )
            else:
                print("MTCNN is unavailable. Falling back to OpenCV Haar Cascade.")

    def detect_main_face(self, img_rgb):
        if self.detector is not None:
            results = self.detector.detect_faces(img_rgb)
            if not results:
                return None
            return max(results, key=lambda face: face["box"][2] * face["box"][3])

        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(
            img_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )
        if len(faces) == 0:
            return None

        x, y, width, height = max(faces, key=lambda face: face[2] * face[3])
        face_gray = img_gray[y : y + height, x : x + width]
        eyes = self.eye_cascade.detectMultiScale(
            face_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(10, 10),
        )

        keypoints = {}
        if len(eyes) >= 2:
            eye_centers = [
                (x + ex + ew // 2, y + ey + eh // 2)
                for ex, ey, ew, eh in sorted(eyes, key=lambda eye: eye[0])[:2]
            ]
            keypoints = {
                "left_eye": eye_centers[0],
                "right_eye": eye_centers[1],
            }

        return {
            "box": [int(x), int(y), int(width), int(height)],
            "keypoints": keypoints,
        }

    def align_face(self, img, left_eye, right_eye):
        d_y = right_eye[1] - left_eye[1]
        d_x = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(d_y, d_x))

        eyes_center = (
            int((left_eye[0] + right_eye[0]) / 2),
            int((left_eye[1] + right_eye[1]) / 2),
        )

        rotation_matrix = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
        height, width = img.shape[:2]
        return cv2.warpAffine(
            img,
            rotation_matrix,
            (width, height),
            flags=cv2.INTER_CUBIC,
        )

    def process_image(self, image_path):
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        main_face = self.detect_main_face(img_rgb)
        if main_face is None:
            print(f"No face found: {image_path}")
            return None

        keypoints = main_face["keypoints"]
        if "left_eye" in keypoints and "right_eye" in keypoints:
            aligned_img = self.align_face(
                img_rgb,
                keypoints["left_eye"],
                keypoints["right_eye"],
            )
        else:
            aligned_img = img_rgb

        aligned_face = self.detect_main_face(aligned_img)
        if aligned_face is None:
            print(f"No face found after alignment: {image_path}")
            return None

        x, y, width, height = aligned_face["box"]

        x1 = max(0, x - self.margin)
        y1 = max(0, y - self.margin)
        x2 = min(aligned_img.shape[1], x + width + self.margin)
        y2 = min(aligned_img.shape[0], y + height + self.margin)

        cropped_face = aligned_img[y1:y2, x1:x2]
        if cropped_face.size == 0:
            print(f"Empty crop: {image_path}")
            return None

        resized_face = cv2.resize(cropped_face, self.target_size)
        return resized_face.astype("float32") / 255.0

    def save_processed_image(self, image_path, output_path):
        processed_face = self.process_image(image_path)
        if processed_face is None:
            return False

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_img = (processed_face * 255).clip(0, 255).astype(np.uint8)
        output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
        return cv2.imwrite(str(output_path), output_img)

    def process_folder(self, input_dir, output_dir, overwrite=False, limit=None):
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        image_paths = [
            path
            for path in input_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        image_paths.sort()

        if limit is not None:
            image_paths = image_paths[:limit]

        total = len(image_paths)
        saved = 0
        skipped = 0
        failed = 0

        for index, image_path in enumerate(image_paths, start=1):
            relative_path = image_path.relative_to(input_dir)
            output_path = output_dir / relative_path

            if output_path.exists() and not overwrite:
                skipped += 1
                continue

            try:
                ok = self.save_processed_image(image_path, output_path)
            except Exception as exc:
                ok = False
                print(f"Failed: {image_path} ({exc})")

            if ok:
                saved += 1
            else:
                failed += 1

            if index % 100 == 0 or index == total:
                print(
                    f"[{index}/{total}] saved={saved} skipped={skipped} failed={failed}"
                )

        return {
            "total": total,
            "saved": saved,
            "skipped": skipped,
            "failed": failed,
            "output_dir": str(output_dir),
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Align, crop, resize, and normalize face images in a folder."
    )
    parser.add_argument(
        "--input",
        default="faces",
        help="Input image folder. Default: data/faces",
    )
    parser.add_argument(
        "--output",
        default="faces_processed",
        help="Output folder. Default: data/faces_processed",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=224,
        help="Output image width and height. Default: 224",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=15,
        help="Face crop margin in pixels. Default: 15",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output images.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N images. Useful for testing.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    preprocessor = FacePreprocessor(
        target_size=(args.size, args.size),
        margin=args.margin,
    )
    summary = preprocessor.process_folder(
        input_dir=args.input,
        output_dir=args.output,
        overwrite=args.overwrite,
        limit=args.limit,
    )
    print("Done:", summary)


if __name__ == "__main__":
    main()
