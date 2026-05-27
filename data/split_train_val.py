import random
from collections import defaultdict
from pathlib import Path


INPUT_TRAIN = Path("data/train.txt")
OUTPUT_TRAIN_NEW = Path("data/train_new.txt")
OUTPUT_TEST_AG = Path("data/ageg_val.txt")

TEST_RATIO = 0.30
SEED = 42


def parse_line(line: str):
    parts = line.strip().split()
    if len(parts) != 4:
        return None

    path, age, gender, emo = parts
    try:
        age = float(age)
        gender = int(gender)
        emo = int(emo)
    except ValueError:
        return None
    return path, age, gender, emo


def age_bucket(age: float) -> str:
    age = int(age)
    if age < 13:
        return "child"
    if age < 20:
        return "teen"
    if age < 35:
        return "young_adult"
    if age < 60:
        return "adult"
    return "senior"


def stratify_key(age: float, gender: int) -> tuple[str, int]:
    return age_bucket(age), gender


def write_rows(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("path\tage\tgender\temotion\n")
        for row in rows:
            f.write(f"{row[0]}\t{row[1]:g}\t{row[2]}\t{row[3]}\n")


def main():
    rng = random.Random(SEED)

    lines_keep = []
    age_gender_groups = defaultdict(list)

    with INPUT_TRAIN.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw or raw.startswith("#") or raw.lower().startswith("path"):
                continue

            item = parse_line(raw)
            if item is None:
                print(f"[SKIP] Invalid row: {raw}")
                continue

            path, age, gender, emo = item
            if age != -1 and gender != -1:
                age_gender_groups[stratify_key(age, gender)].append(
                    (path, age, gender, -1)
                )
            else:
                lines_keep.append((path, age, gender, emo))

    test_samples = []
    train_samples = []
    for key, samples in age_gender_groups.items():
        rng.shuffle(samples)
        n_test = int(len(samples) * TEST_RATIO)
        if len(samples) > 1 and n_test == 0:
            n_test = 1
        test_samples.extend(samples[:n_test])
        train_samples.extend(samples[n_test:])
        print(
            f"[INFO] Group {key}: total={len(samples)} train={len(samples[n_test:])} "
            f"val={len(samples[:n_test])}"
        )

    train_rows = lines_keep + train_samples
    write_rows(OUTPUT_TEST_AG, test_samples)
    write_rows(OUTPUT_TRAIN_NEW, train_rows)

    print(f"[OK] Total age+gender pool: {sum(len(v) for v in age_gender_groups.values())}")
    print(f"[OK] Wrote {len(test_samples)} rows to {OUTPUT_TEST_AG}")
    print(f"[OK] Wrote {len(train_rows)} rows to {OUTPUT_TRAIN_NEW}")


if __name__ == "__main__":
    main()
