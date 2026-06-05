import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any


RATING_COLUMNS = [
    "evaluator_id",
    "relevance_1_to_5",
    "faithfulness_1_to_5",
    "clarity_usefulness_1_to_5",
    "overall_preference_1_to_5",
    "fabricated_content_flag",
    "qualitative_notes",
]

OUTPUT_COLUMNS = [
    "sample_label",
    "pair_id",
    "split",
    "resume_id",
    "job_id",
    "pairing_strategy",
    "resume_text",
    "job_description",
    "generated_rewrite",
    *RATING_COLUMNS,
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path} line {line_number} is invalid JSON: {error}") from error
        if not isinstance(row, dict):
            raise ValueError(f"{path} line {line_number} must contain a JSON object")
        rows.append(row)
    return rows


def split_instruction_input(value: str) -> tuple[str, str]:
    resume_marker = "RESUME:\n"
    job_marker = "\n\nJOB_DESCRIPTION:\n"
    if resume_marker not in value or job_marker not in value:
        return value.strip(), ""
    resume_part, job_part = value.split(job_marker, 1)
    return resume_part.replace(resume_marker, "", 1).strip(), job_part.strip()


def format_suggestions(suggestions: Any) -> str:
    if isinstance(suggestions, str):
        return suggestions.strip()
    if not isinstance(suggestions, list):
        return ""

    lines: list[str] = []
    for index, suggestion in enumerate(suggestions, start=1):
        if isinstance(suggestion, dict):
            section = suggestion.get("section", "")
            action = suggestion.get("action", "")
            text = suggestion.get("suggestion", "")
            evidence = suggestion.get("evidence_from_resume", "")
            requirement = suggestion.get("job_requirement_addressed", "")
            first_line = f"{index}. [{section}/{action}] {text}".strip()
            lines.append(first_line)
            if evidence:
                lines.append(f"   Evidence: {evidence}")
            if requirement:
                lines.append(f"   Job requirement: {requirement}")
        else:
            lines.append(f"{index}. {suggestion}")
    return "\n".join(lines)


def output_by_pair_id(path: Path) -> dict[str, dict[str, Any]]:
    outputs: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        pair_id = row.get("pair_id")
        if isinstance(pair_id, str):
            outputs[pair_id] = row
    return outputs


def candidate_rows(instructions_path: Path, outputs_path: Path, sample_label: str) -> list[dict[str, str]]:
    outputs = output_by_pair_id(outputs_path)
    rows: list[dict[str, str]] = []
    for instruction in read_jsonl(instructions_path):
        metadata = instruction.get("metadata")
        if not isinstance(metadata, dict):
            continue
        pair_id = metadata.get("pair_id")
        if not isinstance(pair_id, str) or pair_id not in outputs:
            continue
        output = outputs[pair_id]
        parsed_output = output.get("parsed_output")
        if not isinstance(parsed_output, dict):
            continue
        resume_text, job_description = split_instruction_input(str(instruction.get("input", "")))
        row = {
            "sample_label": sample_label,
            "pair_id": pair_id,
            "split": str(metadata.get("split", "")),
            "resume_id": str(metadata.get("resume_id", "")),
            "job_id": str(metadata.get("job_id", "")),
            "pairing_strategy": str(metadata.get("pairing_strategy", "")),
            "resume_text": resume_text,
            "job_description": job_description,
            "generated_rewrite": format_suggestions(parsed_output.get("resume_suggestions")),
        }
        for column in RATING_COLUMNS:
            row[column] = ""
        rows.append(row)
    return rows


def balanced_sample(rows: list[dict[str, str]], sample_size: int, seed: int) -> list[dict[str, str]]:
    if sample_size >= len(rows):
        return sorted(rows, key=lambda row: row["pair_id"])

    randomizer = random.Random(seed)
    by_strategy: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_strategy.setdefault(row["pairing_strategy"], []).append(row)

    selected: list[dict[str, str]] = []
    for strategy in sorted(by_strategy):
        if len(selected) >= sample_size:
            break
        bucket = sorted(by_strategy[strategy], key=lambda row: row["pair_id"])
        selected.append(randomizer.choice(bucket))

    remaining = [row for row in rows if row not in selected]
    randomizer.shuffle(remaining)
    selected.extend(remaining[: max(0, sample_size - len(selected))])
    return sorted(selected[:sample_size], key=lambda row: row["pair_id"])


def build_manual_evaluation_rows(
    instructions_path: Path,
    outputs_path: Path,
    sample_size: int,
    seed: int,
    sample_label: str,
) -> list[dict[str, str]]:
    rows = candidate_rows(instructions_path, outputs_path, sample_label)
    return balanced_sample(rows, sample_size, seed)


def write_manual_evaluation_artifacts(rows: list[dict[str, str]], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "manual_evaluation_form.csv"
    markdown_path = output_dir / "manual_evaluation_packet.md"

    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Phase 7 Manual Evaluation Packet",
        "",
        "Rate each generated rewrite from 1 to 5 for relevance, faithfulness, clarity/usefulness, and overall preference. Mark fabricated_content_flag as yes if any suggestion appears to add experience not supported by the original resume.",
        "",
    ]
    for index, row in enumerate(rows, start=1):
        lines.extend(
            [
                f"## Example {index}: {row['pair_id']}",
                "",
                f"- Split: {row['split']}",
                f"- Pairing strategy: {row['pairing_strategy']}",
                "",
                "### Original Resume",
                "",
                row["resume_text"],
                "",
                "### Target Job Description",
                "",
                row["job_description"],
                "",
                "### Generated Rewrite Suggestions",
                "",
                row["generated_rewrite"],
                "",
                "### Reviewer Scores",
                "",
                "- Relevance to the job (1-5):",
                "- Faithfulness to the original resume (1-5):",
                "- Clarity and usefulness (1-5):",
                "- Overall preference (1-5):",
                "- Fabricated content? yes/no:",
                "- Qualitative notes:",
                "",
            ]
        )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, markdown_path


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Phase 7 manual evaluation artifacts.")
    parser.add_argument("--instructions", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/manual_evaluation"))
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-label", default="validation_sample")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    rows = build_manual_evaluation_rows(
        instructions_path=args.instructions,
        outputs_path=args.outputs,
        sample_size=args.sample_size,
        seed=args.seed,
        sample_label=args.sample_label,
    )
    csv_path, markdown_path = write_manual_evaluation_artifacts(rows, args.output_dir)
    print(f"selected {len(rows)} examples")
    print(f"wrote: {csv_path}")
    print(f"wrote: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
