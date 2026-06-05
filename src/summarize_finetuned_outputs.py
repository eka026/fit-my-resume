import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any


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


def extract_scores(rows: list[dict[str, Any]]) -> list[float]:
    scores: list[float] = []
    for row in rows:
        if not row.get("parse_success"):
            continue
        parsed_output = row.get("parsed_output")
        if not isinstance(parsed_output, dict):
            continue
        score = parsed_output.get("score")
        if isinstance(score, int | float):
            scores.append(float(score))
    return scores


def summarize_outputs(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    parse_success = sum(1 for row in rows if row.get("parse_success") is True)
    parse_failed = len(rows) - parse_success
    scores = extract_scores(rows)

    summary: dict[str, Any] = {
        "rows": len(rows),
        "parse_success": parse_success,
        "parse_failed": parse_failed,
        "parse_rate": round((parse_success / len(rows) * 100), 4) if rows else 0.0,
        "score_count": len(scores),
        "score_min": min(scores) if scores else None,
        "score_max": max(scores) if scores else None,
        "score_mean": round(mean(scores), 4) if scores else None,
    }
    return summary


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize fine-tuned Qwen vLLM JSONL outputs.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Optional path to write the summary JSON.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    summary = summarize_outputs(args.input)
    for key, value in summary.items():
        print(f"{key}: {value}")
    if args.summary_output is not None:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"wrote: {args.summary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
