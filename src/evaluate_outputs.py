import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any


@dataclass(frozen=True)
class TeacherReference:
    pair_id: str
    score: float
    explanation_text: str


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


def extract_explanation_text(explanation: Any) -> str:
    if isinstance(explanation, str):
        return " ".join(explanation.split())
    if not isinstance(explanation, dict):
        return ""

    parts: list[str] = []
    for key in ("matched_qualifications", "missing_or_weak_qualifications"):
        value = explanation.get(key)
        if isinstance(value, list):
            parts.extend(str(item) for item in value if item is not None)
        elif isinstance(value, str):
            parts.append(value)

    reasoning = explanation.get("overall_reasoning")
    if isinstance(reasoning, str):
        parts.append(reasoning)

    return " ".join(" ".join(parts).split())


def load_teacher_references(path: Path) -> dict[str, TeacherReference]:
    references: dict[str, TeacherReference] = {}
    for row in read_jsonl(path):
        pair_id = row.get("pair_id")
        teacher_output = row.get("teacher_output")
        if not isinstance(pair_id, str) or not isinstance(teacher_output, dict):
            continue
        score = teacher_output.get("score")
        if not isinstance(score, int | float):
            continue
        references[pair_id] = TeacherReference(
            pair_id=pair_id,
            score=float(score),
            explanation_text=extract_explanation_text(teacher_output.get("explanation")),
        )
    return references


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in text.split() if token.strip()]


def rouge_l_f1(prediction: str, reference: str) -> float:
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0

    previous = [0] * (len(ref_tokens) + 1)
    for pred_token in pred_tokens:
        current = [0]
        for index, ref_token in enumerate(ref_tokens, start=1):
            if pred_token == ref_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current

    lcs = previous[-1]
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return round((2 * precision * recall / (precision + recall)) * 100, 4)


def pearson_correlation(predictions: list[float], references: list[float]) -> float | None:
    if len(predictions) < 2:
        return None
    pred_mean = mean(predictions)
    ref_mean = mean(references)
    numerator = sum((pred - pred_mean) * (ref - ref_mean) for pred, ref in zip(predictions, references))
    pred_denominator = math.sqrt(sum((pred - pred_mean) ** 2 for pred in predictions))
    ref_denominator = math.sqrt(sum((ref - ref_mean) ** 2 for ref in references))
    if pred_denominator == 0 or ref_denominator == 0:
        return None
    return round((numerator / (pred_denominator * ref_denominator)) * 100, 4)


def rank_values(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(indexed):
        end = position
        while end + 1 < len(indexed) and indexed[end + 1][1] == indexed[position][1]:
            end += 1
        average_rank = (position + end + 2) / 2
        for index in range(position, end + 1):
            ranks[indexed[index][0]] = average_rank
        position = end + 1
    return ranks


def macro_f1_score(predictions: list[int], references: list[int]) -> float:
    labels = sorted(set(predictions) | set(references))
    scores: list[float] = []
    for label in labels:
        true_positive = sum(1 for pred, ref in zip(predictions, references) if pred == label and ref == label)
        false_positive = sum(1 for pred, ref in zip(predictions, references) if pred == label and ref != label)
        false_negative = sum(1 for pred, ref in zip(predictions, references) if pred != label and ref == label)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append(2 * precision * recall / (precision + recall))
    return round(mean(scores) * 100, 4) if scores else 0.0


def prediction_score(row: dict[str, Any]) -> float | None:
    if isinstance(row.get("pred_score"), int | float):
        return float(row["pred_score"])
    parsed_output = row.get("parsed_output")
    if isinstance(parsed_output, dict) and isinstance(parsed_output.get("score"), int | float):
        return float(parsed_output["score"])
    return None


def prediction_explanation(row: dict[str, Any]) -> str:
    parsed_output = row.get("parsed_output")
    if isinstance(parsed_output, dict):
        return extract_explanation_text(parsed_output.get("explanation"))
    raw = row.get("raw_output")
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return raw
        if isinstance(parsed, dict):
            return extract_explanation_text(parsed.get("explanation"))
    return ""


def evaluate_score_predictions(
    rows: list[dict[str, Any]],
    references: dict[str, float],
    threshold: float,
) -> dict[str, Any]:
    predictions: list[float] = []
    gold: list[float] = []
    for row in rows:
        pair_id = row.get("pair_id")
        if not isinstance(pair_id, str) or pair_id not in references:
            continue
        score = prediction_score(row)
        if score is None:
            continue
        predictions.append(score)
        gold.append(float(references[pair_id]))

    if not predictions:
        return {
            "evaluated_count": 0,
            "mae": None,
            "rmse": None,
            "pearson": None,
            "spearman": None,
            "accuracy": None,
            "macro_f1": None,
        }

    pred_binary = [1 if score >= threshold else 0 for score in predictions]
    gold_binary = [1 if score >= threshold else 0 for score in gold]
    correct = sum(1 for pred, ref in zip(pred_binary, gold_binary) if pred == ref)
    squared_errors = [(pred - ref) ** 2 for pred, ref in zip(predictions, gold)]

    return {
        "evaluated_count": len(predictions),
        "mae": round(mean(abs(pred - ref) for pred, ref in zip(predictions, gold)), 4),
        "rmse": round(math.sqrt(mean(squared_errors)), 4),
        "pearson": pearson_correlation(predictions, gold),
        "spearman": pearson_correlation(rank_values(predictions), rank_values(gold)),
        "accuracy": round(correct / len(predictions) * 100, 4),
        "macro_f1": macro_f1_score(pred_binary, gold_binary),
    }


def evaluate_explanations(
    rows: list[dict[str, Any]],
    references: dict[str, TeacherReference],
) -> dict[str, Any]:
    scores: list[float] = []
    for row in rows:
        pair_id = row.get("pair_id")
        if not isinstance(pair_id, str) or pair_id not in references:
            continue
        prediction = prediction_explanation(row)
        reference = references[pair_id].explanation_text
        if prediction and reference:
            scores.append(rouge_l_f1(prediction, reference))
    return {
        "evaluated_count": len(scores),
        "rouge_l_f1": round(mean(scores), 4) if scores else None,
    }


def parse_rate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"rows": 0, "parse_success": 0, "parse_rate": 0.0}
    if all("parse_success" in row for row in rows):
        successes = sum(1 for row in rows if row.get("parse_success") is True)
    else:
        successes = sum(1 for row in rows if prediction_score(row) is not None)
    return {
        "rows": len(rows),
        "parse_success": successes,
        "parse_rate": round(successes / len(rows) * 100, 4),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "system",
        "score_evaluated_count",
        "mae",
        "rmse",
        "pearson",
        "spearman",
        "accuracy",
        "macro_f1",
        "explanation_evaluated_count",
        "rouge_l_f1",
    ]
    lines = [
        "# Phase 6 Automatic Evaluation Summary",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_systems(
    teacher_path: Path,
    system_paths: dict[str, Path],
    output_dir: Path,
    threshold: float,
) -> list[dict[str, Any]]:
    references = load_teacher_references(teacher_path)
    reference_scores = {pair_id: reference.score for pair_id, reference in references.items()}
    summary_rows: list[dict[str, Any]] = []
    parse_rows: list[dict[str, Any]] = []

    for system_name, path in system_paths.items():
        rows = read_jsonl(path)
        score_metrics = evaluate_score_predictions(rows, reference_scores, threshold)
        explanation_metrics = evaluate_explanations(rows, references)
        parse_metrics = parse_rate(rows)
        summary_rows.append(
            {
                "system": system_name,
                **{
                    ("score_evaluated_count" if key == "evaluated_count" else key): value
                    for key, value in score_metrics.items()
                },
                **{
                    ("explanation_evaluated_count" if key == "evaluated_count" else key): value
                    for key, value in explanation_metrics.items()
                },
            }
        )
        parse_rows.append({"system": system_name, **parse_metrics})

    write_csv(output_dir / "score_and_explanation_metrics.csv", summary_rows)
    write_csv(output_dir / "parse_metrics.csv", parse_rows)
    write_summary_markdown(output_dir / "phase6_summary.md", summary_rows)
    return summary_rows


def parse_system_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("System inputs must use name=path format")
    name, path = value.split("=", 1)
    if not name.strip():
        raise argparse.ArgumentTypeError("System name cannot be empty")
    return name.strip(), Path(path)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FitMyResume Phase 6 outputs.")
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--system", action="append", type=parse_system_arg, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/evaluation"))
    parser.add_argument("--threshold", type=float, default=50.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    systems = dict(args.system)
    rows = evaluate_systems(args.teacher, systems, args.output_dir, args.threshold)
    for row in rows:
        print(
            f"{row['system']}: n={row['score_evaluated_count']} "
            f"mae={row['mae']} rouge_l_f1={row['rouge_l_f1']}"
        )
    print(f"wrote: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
