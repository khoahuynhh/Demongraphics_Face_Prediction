from pathlib import Path


DATA_DIR = Path("data")
VAL_EMO_ROOT = DATA_DIR / "test"
PREFERRED_VAL_EMO_ROOT = None
OUTPUT_TXT = DATA_DIR / "emo_val.txt"

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


def resolve_rel_path(rel_path: Path) -> str:
    candidates = []
    if PREFERRED_VAL_EMO_ROOT is not None:
        candidates.append(DATA_DIR / PREFERRED_VAL_EMO_ROOT.name / rel_path)
    candidates.append(DATA_DIR / rel_path)

    for candidate in candidates:
        if candidate.exists():
            return str(candidate.relative_to(DATA_DIR)).replace("\\", "/")
    return str(rel_path).replace("\\", "/")


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
                    resolve_rel_path(rel_path),
                    age,
                    gender,
                    emo_id,
                ]
            )
    return rows


def write_rows(output_path: Path, rows):
    with output_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("path\tage\tgender\temotion\n")
        for path, age, gender, emotion in rows:
            f.write(f"{path}\t{age}\t{gender}\t{emotion}\n")


def main():
    rows = collect_emotion_dataset(VAL_EMO_ROOT)
    write_rows(OUTPUT_TXT, rows)
    print(f"[OK] Wrote {len(rows)} rows to {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
