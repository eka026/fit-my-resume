import argparse
import csv
import sys
from pathlib import Path
from statistics import mean
from typing import Any


RATING_COLUMNS = [
    "relevance_1_to_5",
    "faithfulness_1_to_5",
    "clarity_usefulness_1_to_5",
    "overall_preference_1_to_5",
]

TRUE_VALUES = {"yes", "true", "y", "1"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def parse_rating(value: str, column: str, pair_id: str) -> int | None:
    if value.strip() == "":
        return None
    try:
        rating = int(value)
    except ValueError as error:
        raise ValueError(f"{column} for {pair_id} must be an integer from 1 to 5") from error
    if rating < 1 or rating > 5:
        raise ValueError(f"{column} for {pair_id} must be an integer from 1 to 5")
    return rating


def is_rated(row: dict[str, str]) -> bool:
    return any(row.get(column, "").strip() for column in RATING_COLUMNS)


def summarize_manual_evaluation(path: Path) -> dict[str, Any]:
    rows = read_csv(path)
    ratings: dict[str, list[int]] = {column: [] for column in RATING_COLUMNS}
    rated_rows = 0
    evaluator_ids: set[str] = set()
    pair_ids = {row.get("pair_id", "") for row in rows if row.get("pair_id", "")}
    fabrication_flags = 0
    qualitative_issues: list[str] = []

    for row in rows:
        pair_id = row.get("pair_id", "")
        if row.get("fabricated_content_flag", "").strip().lower() in TRUE_VALUES:
            fabrication_flags += 1
        notes = row.get("qualitative_notes", "").strip()
        if notes:
            qualitative_issues.append(notes)

        if not is_rated(row):
            continue

        rated_rows += 1
        evaluator_id = row.get("evaluator_id", "").strip()
        if evaluator_id:
            evaluator_ids.add(evaluator_id)
        for column in RATING_COLUMNS:
            rating = parse_rating(row.get(column, ""), column, pair_id)
            if rating is None:
                raise ValueError(f"{column} for {pair_id} is required when a row has ratings")
            ratings[column].append(rating)

    summary: dict[str, Any] = {
        "source_path": str(path),
        "total_rows": len(rows),
        "rated_rows": rated_rows,
        "unique_examples": len(pair_ids),
        "unique_evaluators": len(evaluator_ids),
        "fabrication_flags": fabrication_flags,
        "common_qualitative_issues": qualitative_issues,
    }
    for column in RATING_COLUMNS:
        output_key = f"average_{column}"
        summary[output_key] = round(mean(ratings[column]), 4) if ratings[column] else None
    return summary


def write_manual_evaluation_summary(summary: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "manual_evaluation_summary_template.csv"
    markdown_path = output_dir / "phase7_manual_evaluation_summary.md"

    flat_summary = {
        key: value
        for key, value in summary.items()
        if key != "common_qualitative_issues"
    }
    flat_summary["common_qualitative_issues"] = " | ".join(summary.get("common_qualitative_issues", []))

    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(flat_summary))
        writer.writeheader()
        writer.writerow(flat_summary)

    lines = [
        "# Phase 7 Manual Evaluation Summary",
        "",
        f"Source form: {summary['source_path']}",
        "",
        f"- Total rows: {summary['total_rows']}",
        f"- Rated rows: {summary['rated_rows']}",
        f"- Unique examples: {summary['unique_examples']}",
        f"- Unique evaluators: {summary['unique_evaluators']}",
        f"- Fabrication flags: {summary['fabrication_flags']}",
        "",
        "## Average Scores",
        "",
        f"- Relevance to the job: {summary['average_relevance_1_to_5']}",
        f"- Faithfulness to the original resume: {summary['average_faithfulness_1_to_5']}",
        f"- Clarity and usefulness: {summary['average_clarity_usefulness_1_to_5']}",
        f"- Overall preference: {summary['average_overall_preference_1_to_5']}",
        "",
        "## Common Qualitative Issues",
        "",
    ]
    issues = summary.get("common_qualitative_issues", [])
    if issues:
        lines.extend(f"- {issue}" for issue in issues)
    else:
        lines.append("No human qualitative notes have been entered yet.")
    if summary["rated_rows"] == 0:
        lines.extend(
            [
                "",
                "## Current Status",
                "",
                "The manual evaluation packet has been prepared, but human ratings have not been entered yet. Fill `manual_evaluation_form.csv` and rerun `src/summarize_manual_evaluation.py` before making final manual-evaluation claims.",
            ]
        )

    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, markdown_path


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize completed Phase 7 manual evaluation ratings.")
    parser.add_argument("--form", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/manual_evaluation"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    summary = summarize_manual_evaluation(args.form)
    csv_path, markdown_path = write_manual_evaluation_summary(summary, args.output_dir)
    print(f"rated rows: {summary['rated_rows']}")
    print(f"wrote: {csv_path}")
    print(f"wrote: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
