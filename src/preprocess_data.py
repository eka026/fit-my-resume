import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd


SPLIT_FRACTIONS = {"train": 0.8, "validation": 0.1, "test": 0.1}
RESUME_COLUMNS = ["ID", "Resume_str", "Category"]
JOB_COLUMNS = [
    "company_name",
    "job_description",
    "position_title",
    "description_length",
    "model_response",
]

TEXT_REPLACEMENTS = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u00a0": " ",
    }
)

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
URL_RE = re.compile(r"\b(?:https?://|www\.)\S+\b", re.IGNORECASE)
PHONE_RE = re.compile(
    r"(?<!\w)(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}(?!\w)"
)
ADDRESS_LINE_RE = re.compile(
    r"^\s*\d{1,6}\s+.+\b(?:street|st\.?|avenue|ave\.?|road|rd\.?|drive|dr\.?|lane|ln\.?|boulevard|blvd\.?)\b.*$",
    re.IGNORECASE,
)


def clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""

    text = str(value).translate(TEXT_REPLACEMENTS)
    text = "".join(
        char if char in "\n\r\t" or not _is_control_character(char) else " "
        for char in text
    )
    return re.sub(r"\s+", " ", text).strip()


def mask_resume_pii(text: str) -> str:
    masked_lines = []
    for line in str(text).splitlines():
        if ADDRESS_LINE_RE.match(line):
            masked_lines.append("[ADDRESS]")
        else:
            masked_lines.append(line)

    masked = "\n".join(masked_lines)
    masked = EMAIL_RE.sub("[EMAIL]", masked)
    masked = URL_RE.sub("[URL]", masked)
    masked = PHONE_RE.sub("[PHONE]", masked)
    return masked


def split_dataframe(
    df: pd.DataFrame, id_column: str, seed: int
) -> dict[str, pd.DataFrame]:
    sorted_df = df.sort_values(id_column, kind="mergesort").reset_index(drop=True)
    indices = list(sorted_df.index)
    random.Random(seed).shuffle(indices)

    n_rows = len(indices)
    train_end = int(n_rows * SPLIT_FRACTIONS["train"])
    validation_end = int(n_rows * (SPLIT_FRACTIONS["train"] + SPLIT_FRACTIONS["validation"]))

    split_indices = {
        "train": indices[:train_end],
        "validation": indices[train_end:validation_end],
        "test": indices[validation_end:],
    }

    return {
        name: sorted_df.loc[rows].reset_index(drop=True)
        for name, rows in split_indices.items()
    }


def preprocess_resumes(raw_dir: Path) -> tuple[pd.DataFrame, int]:
    path = raw_dir / "Resume.csv"
    raw = _read_csv_with_required_columns(path, RESUME_COLUMNS)
    input_rows = len(raw)

    cleaned = pd.DataFrame(
        {
            "resume_id": raw["ID"],
            "category": raw["Category"].map(clean_text),
            "resume_text": raw["Resume_str"].map(mask_resume_pii).map(clean_text),
        }
    )
    cleaned = cleaned[cleaned["resume_text"] != ""].copy()
    cleaned = cleaned.sort_values("resume_id", kind="mergesort")
    cleaned = cleaned.drop_duplicates(subset=["resume_text"], keep="first")
    cleaned = cleaned[["resume_id", "category", "resume_text"]].reset_index(drop=True)
    return cleaned, input_rows


def preprocess_jobs(raw_dir: Path) -> tuple[pd.DataFrame, int]:
    path = raw_dir / "training_data.csv"
    raw = _read_csv_with_required_columns(path, JOB_COLUMNS)
    input_rows = len(raw)

    cleaned = raw.copy()
    for column in ["company_name", "position_title", "job_description", "model_response"]:
        cleaned[column] = cleaned[column].map(clean_text)

    cleaned = cleaned[cleaned["job_description"] != ""].copy()
    cleaned = cleaned.sort_values(
        ["position_title", "company_name", "job_description"],
        kind="mergesort",
    )
    cleaned = cleaned.drop_duplicates(subset=["job_description"], keep="first")
    cleaned = cleaned.reset_index(drop=True)
    cleaned.insert(0, "job_id", [f"job_{index + 1:06d}" for index in range(len(cleaned))])
    cleaned = cleaned[
        [
            "job_id",
            "company_name",
            "position_title",
            "description_length",
            "job_description",
            "model_response",
        ]
    ]
    return cleaned, input_rows


def write_outputs(
    resumes: pd.DataFrame,
    jobs: pd.DataFrame,
    resume_input_rows: int,
    job_input_rows: int,
    raw_dir: Path,
    interim_dir: Path,
    processed_dir: Path,
    seed: int,
) -> None:
    interim_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    output_counts: dict[str, int] = {}

    _write_csv(resumes, interim_dir / "resumes_clean.csv", output_counts)
    _write_csv(jobs, interim_dir / "jobs_clean.csv", output_counts)

    resume_splits = split_dataframe(resumes, id_column="resume_id", seed=seed)
    job_splits = split_dataframe(jobs, id_column="job_id", seed=seed)

    for split_name, split_df in resume_splits.items():
        _write_csv(split_df, processed_dir / f"resumes_{split_name}.csv", output_counts)

    for split_name, split_df in job_splits.items():
        _write_csv(split_df, processed_dir / f"jobs_{split_name}.csv", output_counts)

    metadata_path = processed_dir / "preprocessing_metadata.json"
    output_counts[str(metadata_path)] = 1

    metadata = {
        "input_files": {
            "Resume.csv": {
                "path": str(raw_dir / "Resume.csv"),
                "required_columns": RESUME_COLUMNS,
                "rows": resume_input_rows,
            },
            "training_data.csv": {
                "path": str(raw_dir / "training_data.csv"),
                "required_columns": JOB_COLUMNS,
                "rows": job_input_rows,
            },
        },
        "output_files": output_counts,
        "script": "src/preprocess_data.py",
        "seed": seed,
        "split_fractions": SPLIT_FRACTIONS,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess FitMyResume raw Kaggle data.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/kaggle"))
    parser.add_argument("--interim-dir", type=Path, default=Path("data/interim"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    resumes, resume_input_rows = preprocess_resumes(args.raw_dir)
    jobs, job_input_rows = preprocess_jobs(args.raw_dir)
    write_outputs(
        resumes=resumes,
        jobs=jobs,
        resume_input_rows=resume_input_rows,
        job_input_rows=job_input_rows,
        raw_dir=args.raw_dir,
        interim_dir=args.interim_dir,
        processed_dir=args.processed_dir,
        seed=args.seed,
    )

    print(f"Cleaned resumes: {len(resumes)}")
    print(f"Cleaned jobs: {len(jobs)}")
    print(f"Wrote interim files to: {args.interim_dir}")
    print(f"Wrote processed files to: {args.processed_dir}")
    return 0


def _read_csv_with_required_columns(path: Path, required_columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input file: {path}")

    df = pd.read_csv(path)
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
    return df


def _write_csv(df: pd.DataFrame, path: Path, output_counts: dict[str, int]) -> None:
    df.to_csv(path, index=False)
    output_counts[str(path)] = len(df)


def _is_control_character(char: str) -> bool:
    return ord(char) < 32 or ord(char) == 127


if __name__ == "__main__":
    raise SystemExit(main())
