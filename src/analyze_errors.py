import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


FAILURE_MODES = [
    "Invalid JSON",
    "Wrong or poorly calibrated score",
    "Missing important job requirements",
    "Overly generic explanation",
    "Fabricated resume content risk",
    "Rewrite that removes important candidate evidence",
    "Prompt injection vulnerability",
]

SYSTEM_COLUMNS = ["bm25", "sentence_transf", "zero_shot", "finetuned_v2"]
GENERIC_EXPLANATION_TERMS = (
    "good fit",
    "strong fit",
    "may be a good",
    "some skills",
    "relevant experience",
    "the role",
)


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


def numeric(value: Any) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None


def sample_weak_score_examples(
    rows: list[dict[str, Any]],
    systems: list[str],
    limit: int,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for system in systems:
        system_errors: list[dict[str, Any]] = []
        for row in rows:
            teacher_score = numeric(row.get("teacher_score"))
            pred_score = numeric(row.get(system))
            pair_id = row.get("pair_id")
            if teacher_score is None or pred_score is None or not isinstance(pair_id, str):
                continue
            system_errors.append(
                {
                    "system": system,
                    "pair_id": pair_id,
                    "strategy": str(row.get("strategy") or row.get("pairing_strategy") or ""),
                    "teacher_score": teacher_score,
                    "pred_score": pred_score,
                    "absolute_error": round(abs(pred_score - teacher_score), 4),
                }
            )
        samples.extend(
            sorted(system_errors, key=lambda item: (-item["absolute_error"], item["pair_id"]))[:limit]
        )
    return samples


def parsed_output(row: dict[str, Any]) -> dict[str, Any]:
    output = row.get("parsed_output")
    if isinstance(output, dict):
        return output
    raw = row.get("raw_response") or row.get("raw_output")
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def has_invalid_raw_json(row: dict[str, Any]) -> bool:
    raw = row.get("raw_response") or row.get("raw_output")
    if not isinstance(raw, str) or not raw.strip():
        return False
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return True
    return not isinstance(parsed, dict)


def text_from_explanation(explanation: Any) -> str:
    if isinstance(explanation, str):
        return explanation
    if not isinstance(explanation, dict):
        return ""
    parts: list[str] = []
    for value in explanation.values():
        if isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif isinstance(value, str):
            parts.append(value)
    return " ".join(parts)


def suggestion_rows(output: dict[str, Any]) -> list[dict[str, Any]]:
    suggestions = output.get("resume_suggestions")
    if isinstance(suggestions, list):
        return [item for item in suggestions if isinstance(item, dict)]
    return []


def has_prompt_injection_text(row: dict[str, Any]) -> bool:
    haystack = " ".join(
        str(row.get(key, ""))
        for key in ("input", "resume_text", "job_description", "raw_response", "raw_output")
    ).lower()
    return bool(re.search(r"ignore (all )?(previous|prior) instructions|system prompt|developer message", haystack))


def categorize_model_output(row: dict[str, Any], score_error_threshold: float = 25.0) -> dict[str, str]:
    modes: list[str] = []
    if row.get("parse_success") is False or has_invalid_raw_json(row):
        modes.append("Invalid JSON")

    output = parsed_output(row)
    score = numeric(output.get("score"))
    teacher_score = numeric(row.get("teacher_score") or row.get("gt_score"))
    if score is not None and teacher_score is not None and abs(score - teacher_score) >= score_error_threshold:
        modes.append("Wrong or poorly calibrated score")

    explanation_text = text_from_explanation(output.get("explanation")).lower()
    if explanation_text:
        if len(explanation_text.split()) < 18 or any(term in explanation_text for term in GENERIC_EXPLANATION_TERMS):
            modes.append("Overly generic explanation")
        if not any(term in explanation_text for term in ("missing", "lack", "no ", "gap", "weak", "require")):
            modes.append("Missing important job requirements")
    elif output:
        modes.append("Missing important job requirements")

    suggestions = suggestion_rows(output)
    if suggestions and any(
        (
            item.get("action") == "add_if_true"
            or "if true" in str(item.get("suggestion", "")).lower()
            or str(item.get("suggestion", "")).lower().startswith("add ")
        )
        and not str(item.get("evidence_from_resume", "")).strip()
        for item in suggestions
    ):
        modes.append("Fabricated resume content risk")
    if suggestions and any(
        not str(item.get("evidence_from_resume", "")).strip()
        and str(item.get("action", "")).lower() in {"remove", "rewrite", "replace"}
        for item in suggestions
    ):
        modes.append("Rewrite that removes important candidate evidence")
    if has_prompt_injection_text(row):
        modes.append("Prompt injection vulnerability")

    return {
        "pair_id": str(row.get("pair_id", "")),
        "failure_modes": "; ".join(mode for mode in FAILURE_MODES if mode in modes),
        "parse_error": str(row.get("parse_error", "")),
    }


def infer_failure_sources(rows: list[dict[str, Any]]) -> dict[str, int]:
    source_counts: Counter[str] = Counter()
    for row in rows:
        modes = str(row.get("failure_modes", ""))
        if "Invalid JSON" in modes:
            source_counts["Prompt design"] += 1
            source_counts["Compute-constrained truncation"] += 1
        if "Wrong or poorly calibrated score" in modes:
            source_counts["Teacher-output quality"] += 1
            source_counts["Model limitations"] += 1
        if "Missing important job requirements" in modes or "Overly generic explanation" in modes:
            source_counts["Prompt design"] += 1
            source_counts["Model limitations"] += 1
        if "Fabricated resume content risk" in modes:
            source_counts["Dataset noise"] += 1
            source_counts["Prompt design"] += 1
        if "Rewrite that removes important candidate evidence" in modes:
            source_counts["Training settings"] += 1
            source_counts["Model limitations"] += 1
        if "Prompt injection vulnerability" in modes:
            source_counts["Prompt design"] += 1
    return dict(sorted(source_counts.items()))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    path: Path,
    score_samples: list[dict[str, Any]],
    qualitative_samples: list[dict[str, Any]],
    source_counts: dict[str, int],
) -> None:
    mode_counts = Counter()
    for row in qualitative_samples:
        for mode in str(row.get("failure_modes", "")).split("; "):
            if mode:
                mode_counts[mode] += 1

    lines = [
        "# Phase 8 Error Analysis Summary",
        "",
        "## Scope",
        "",
        "This analysis uses saved project artifacts rather than rerunning generation. Score weak cases come from `results/all_methods_test_results.jsonl`; qualitative failure labels come from saved validation outputs with parse status and generated JSON.",
        "",
        "## Weak Score Examples",
        "",
        "| system | pair_id | teacher_score | pred_score | absolute_error |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in score_samples[:20]:
        lines.append(
            f"| {row['system']} | {row['pair_id']} | {row['teacher_score']} | "
            f"{row['pred_score']} | {row['absolute_error']} |"
        )

    lines.extend(["", "## Failure Mode Counts", "", "| failure_mode | count |", "| --- | --- |"])
    for mode in FAILURE_MODES:
        lines.append(f"| {mode} | {mode_counts.get(mode, 0)} |")

    lines.extend(["", "## Likely Failure Sources", "", "| source | count |", "| --- | --- |"])
    for source, count in source_counts.items():
        lines.append(f"| {source} | {count} |")

    lines.extend(
        [
            "",
            "## Iteration Notes",
            "",
            "- No second fine-tuning run was performed in this phase because the current artifacts are sufficient for the CS455 final report and additional training would require GPU time.",
            "- Targeted prompt iteration should focus on stricter JSON-only generation, requirement-by-requirement evidence, and requiring every resume suggestion to cite resume evidence or use `add_if_true`.",
            "- Low-quality teacher examples should be removed when the teacher score conflicts with clear resume/job evidence or when suggestions lack evidence.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_phase8_report(
    all_methods_path: Path,
    model_paths: dict[str, Path],
    output_dir: Path,
    score_sample_limit: int,
    qualitative_sample_limit: int,
) -> dict[str, Path]:
    all_methods = read_jsonl(all_methods_path)
    systems = [system for system in SYSTEM_COLUMNS if any(system in row for row in all_methods)]
    score_samples = sample_weak_score_examples(all_methods, systems, score_sample_limit)

    qualitative_samples: list[dict[str, Any]] = []
    for system, path in model_paths.items():
        if not path.exists():
            continue
        system_rows = []
        for row in read_jsonl(path):
            categorized = categorize_model_output(row)
            if categorized["failure_modes"]:
                system_rows.append({"system": system, **categorized})
        qualitative_samples.extend(system_rows[:qualitative_sample_limit])

    source_counts = infer_failure_sources(qualitative_samples)
    score_path = output_dir / "weak_score_examples.csv"
    qualitative_path = output_dir / "qualitative_failure_samples.csv"
    summary_path = output_dir / "phase8_error_analysis_summary.md"
    write_csv(score_path, score_samples)
    write_csv(qualitative_path, qualitative_samples)
    write_markdown(summary_path, score_samples, qualitative_samples, source_counts)
    return {
        "score_samples": score_path,
        "qualitative_samples": qualitative_path,
        "summary": summary_path,
    }


def parse_system_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use name=path for model outputs")
    name, path = value.split("=", 1)
    return name, Path(path)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 8 error analysis.")
    parser.add_argument("--all-methods", type=Path, default=Path("results/all_methods_test_results.jsonl"))
    parser.add_argument("--model-output", action="append", type=parse_system_arg, default=[])
    parser.add_argument("--output-dir", type=Path, default=Path("results/error_analysis"))
    parser.add_argument("--score-sample-limit", type=int, default=10)
    parser.add_argument("--qualitative-sample-limit", type=int, default=25)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    model_paths = dict(args.model_output) or {
        "zero_shot_qwen": Path("results/zero_shot_qwen_val_outputs.jsonl"),
        "finetuned_qwen": Path("results/finetuned_qwen_validation_transformers_sample50_outputs.jsonl"),
    }
    paths = write_phase8_report(
        all_methods_path=args.all_methods,
        model_paths=model_paths,
        output_dir=args.output_dir,
        score_sample_limit=args.score_sample_limit,
        qualitative_sample_limit=args.qualitative_sample_limit,
    )
    for label, path in paths.items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
