import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from src.run_teacher_pilot import validate_teacher_output
except ModuleNotFoundError:
    from run_teacher_pilot import validate_teacher_output


INSTRUCTION = (
    "Evaluate the resume against the job description. Return only valid JSON with "
    "score, explanation, and resume_suggestions. Do not invent experience."
)

ALLOWED_SECTIONS = {
    "summary",
    "skills",
    "experience",
    "education",
    "projects",
    "certifications",
    "other",
}
ALLOWED_ACTIONS = {"emphasize", "reorder", "reword", "remove", "add_if_true"}

PAIR_COLUMNS = {
    "pair_id",
    "split",
    "resume_id",
    "job_id",
    "pairing_strategy",
    "similarity_score",
    "resume_text",
    "job_description",
}


def validate_strict_teacher_output(output: dict[str, Any]) -> None:
    validate_teacher_output(output)
    for index, suggestion in enumerate(output["resume_suggestions"]):
        section = suggestion["section"]
        action = suggestion["action"]
        if section not in ALLOWED_SECTIONS:
            raise ValueError(
                f"resume_suggestions[{index}].section must be one of "
                f"{sorted(ALLOWED_SECTIONS)}; got {section!r}"
            )
        if action not in ALLOWED_ACTIONS:
            raise ValueError(
                f"resume_suggestions[{index}].action must be one of "
                f"{sorted(ALLOWED_ACTIONS)}; got {action!r}"
            )
        evidence = suggestion["evidence_from_resume"].strip()
        if action != "add_if_true" and not evidence:
            raise ValueError(
                f"resume_suggestions[{index}].evidence_from_resume cannot be empty "
                f"when action is {action!r}"
            )


def load_teacher_outputs(path: Path) -> dict[str, dict[str, Any]]:
    outputs: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path} line {line_number} is invalid JSON: {error}") from error

        pair_id = row.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id:
            raise ValueError(f"{path} line {line_number} is missing a valid pair_id")
        if pair_id in outputs:
            raise ValueError(f"{path} contains duplicate pair_id {pair_id!r}")
        if "teacher_output" not in row:
            raise ValueError(f"{path} line {line_number} is missing teacher_output")
        outputs[pair_id] = row
    return outputs


def load_pairs(path: Path) -> pd.DataFrame:
    pairs = pd.read_csv(path)
    missing = sorted(PAIR_COLUMNS - set(pairs.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
    if pairs["pair_id"].duplicated().any():
        duplicates = pairs.loc[pairs["pair_id"].duplicated(), "pair_id"].head(5).tolist()
        raise ValueError(f"{path} contains duplicate pair_id values: {duplicates}")
    return pairs


def build_instruction_row(pair: dict[str, Any], teacher_row: dict[str, Any]) -> dict[str, Any]:
    teacher_output = teacher_row["teacher_output"]
    return {
        "instruction": INSTRUCTION,
        "input": f"RESUME:\n{pair['resume_text']}\n\nJOB_DESCRIPTION:\n{pair['job_description']}",
        "output": json.dumps(teacher_output, ensure_ascii=False, separators=(",", ":")),
        "metadata": {
            "pair_id": str(pair["pair_id"]),
            "split": str(pair["split"]),
            "resume_id": str(pair["resume_id"]),
            "job_id": str(pair["job_id"]),
            "pairing_strategy": str(pair["pairing_strategy"]),
            "similarity_score": float(pair["similarity_score"]),
            "prompt_version": str(teacher_row.get("prompt_version", "")),
            "model": str(teacher_row.get("model", "")),
        },
    }


def build_corpus(
    pairs_path: Path,
    teacher_outputs_path: Path,
    output_path: Path,
    skip_invalid: bool = False,
    skip_missing: bool = False,
) -> dict[str, int]:
    pairs = load_pairs(pairs_path)
    teacher_outputs = load_teacher_outputs(teacher_outputs_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "pairs_read": int(len(pairs)),
        "teacher_outputs_read": int(len(teacher_outputs)),
        "written": 0,
        "missing": 0,
        "invalid": 0,
    }

    with output_path.open("w", encoding="utf-8") as output_file:
        for _, pair_series in pairs.iterrows():
            pair = pair_series.to_dict()
            pair_id = str(pair["pair_id"])
            teacher_row = teacher_outputs.get(pair_id)
            if teacher_row is None:
                summary["missing"] += 1
                if skip_missing:
                    continue
                raise ValueError(f"No teacher output found for pair_id {pair_id!r}")

            try:
                validate_strict_teacher_output(teacher_row["teacher_output"])
            except ValueError as error:
                summary["invalid"] += 1
                if skip_invalid:
                    continue
                raise ValueError(f"Invalid teacher output for pair_id {pair_id!r}: {error}") from error

            row = build_instruction_row(pair, teacher_row)
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            summary["written"] += 1

    return summary


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build instruction-tuning JSONL from teacher outputs.")
    parser.add_argument("--pairs-path", type=Path, required=True)
    parser.add_argument("--teacher-outputs-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="Skip teacher rows that fail strict schema validation instead of failing.",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip pair rows with no matching teacher output instead of failing.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    summary = build_corpus(
        pairs_path=args.pairs_path,
        teacher_outputs_path=args.teacher_outputs_path,
        output_path=args.output_path,
        skip_invalid=args.skip_invalid,
        skip_missing=args.skip_missing,
    )
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"wrote: {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
