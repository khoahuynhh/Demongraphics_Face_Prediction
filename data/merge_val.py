from pathlib import Path


INPUT_FILE_1 = Path("data/ageg_val.txt")
INPUT_FILE_2 = Path("data/emo_val.txt")
OUTPUT_FILE = Path("data/val.txt")


def read_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.lower().startswith("path"):
                continue
            rows.append(line)
    return rows


def merge_txt_files(file1: Path, file2: Path, output_file: Path):
    rows = read_rows(file1) + read_rows(file2)
    with output_file.open("w", encoding="utf-8", newline="\n") as out_f:
        out_f.write("path\tage\tgender\temotion\n")
        for row in rows:
            out_f.write(row + "\n")
    print(f"[OK] Wrote {len(rows)} rows to {output_file}")


if __name__ == "__main__":
    merge_txt_files(INPUT_FILE_1, INPUT_FILE_2, OUTPUT_FILE)
