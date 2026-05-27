from pathlib import Path


DATA_DIR = Path("data")
TRAIN_EMO_ROOT = DATA_DIR / "faces_processed"
AGEG_ROOT = DATA_DIR / "faces_processed"
OUTPUT_TXT = DATA_DIR / "train.txt"

DEFAULT_EMO = -1
DEFAULT_AGE = -1
DEFAULT_GENDER = -1

EMO_MAP = {
    "anger": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "neutral": 4,
    "sad": 5,
    "surprise": 6,
}

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def is_image_file(path: Path) -> bool:
    lower = path.name.lower()
    return lower.endswith(IMAGE_SUFFIXES) or ".jpg.chip" in lower


def parse_age_gender_from_filename(filename: str):
    parts = Path(filename).name.split("_")
    if len(parts) < 2:
        return None, None

    try:
        age = int(parts[0])
        gender = int(parts[1])
    except ValueError:
        return None, None

    if gender not in (0, 1):
        return None, None
    return age, gender


def collect_emotion_dataset(emo_root: Path):
    rows = []
    for emo_folder, emo_id in EMO_MAP.items():
        folder_path = emo_root / emo_folder
        if not folder_path.is_dir():
            print(f"[WARN] Missing emotion folder: {folder_path}")
            continue

        for image_path in sorted(folder_path.iterdir()):
            if not image_path.is_file() or not is_image_file(image_path):
                continue

            age, gender = parse_age_gender_from_filename(image_path.name)
            if age is None:
                age = DEFAULT_AGE
            if gender is None:
                gender = DEFAULT_GENDER

            rel_path = Path(emo_root.name) / emo_folder / image_path.name
            rows.append(
                [
                    str(rel_path).replace("\\", "/"),
                    age,
                    gender,
                    emo_id,
                ]
            )
    return rows


def collect_age_gender_only_dataset(root: Path):
    rows = []
    if root is None:
        return rows
    if not root.is_dir():
        print(f"[WARN] AGEG_ROOT not found: {root}")
        return rows

    for image_path in sorted(root.iterdir()):
        if not image_path.is_file() or not is_image_file(image_path):
            continue

        age, gender = parse_age_gender_from_filename(image_path.name)
        if age is None or gender is None:
            print(f"[SKIP] Cannot parse age/gender: {image_path.name}")
            continue

        rel_path = Path(root.name) / image_path.name
        rows.append([str(rel_path).replace("\\", "/"), age, gender, DEFAULT_EMO])
    return rows


def write_rows(output_path: Path, rows):
    with output_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("path\tage\tgender\temotion\n")
        for path, age, gender, emotion in rows:
            f.write(f"{path}\t{age}\t{gender}\t{emotion}\n")


def main():
    all_rows = []
    all_rows.extend(collect_emotion_dataset(TRAIN_EMO_ROOT))
    all_rows.extend(collect_age_gender_only_dataset(AGEG_ROOT))
    write_rows(OUTPUT_TXT, all_rows)
    print(f"[OK] Wrote {len(all_rows)} rows to {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
