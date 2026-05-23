import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd


PROMPT_VERSION = "teacher_gold_output_prompt_v3"
DEFAULT_MODEL = "deepseek-v4-flash"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

TEACHER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "minimum": 0, "maximum": 100},
        "explanation": {
            "type": "object",
            "properties": {
                "matched_qualifications": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "missing_or_weak_qualifications": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "overall_reasoning": {"type": "string"},
            },
            "required": [
                "matched_qualifications",
                "missing_or_weak_qualifications",
                "overall_reasoning",
            ],
        },
        "resume_suggestions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "section": {"type": "string"},
                    "action": {"type": "string"},
                    "suggestion": {"type": "string"},
                    "evidence_from_resume": {"type": "string"},
                    "job_requirement_addressed": {"type": "string"},
                },
                "required": [
                    "section",
                    "action",
                    "suggestion",
                    "evidence_from_resume",
                    "job_requirement_addressed",
                ],
            },
        },
    },
    "required": ["score", "explanation", "resume_suggestions"],
}


def render_prompt(template: str, resume_text: str, job_description: str) -> str:
    rendered = template.replace("{{resume_text}}", resume_text)
    rendered = rendered.replace("{{job_description}}", job_description)
    if "{{resume_text}}" in rendered or "{{job_description}}" in rendered:
        raise ValueError("Prompt still contains unfilled placeholders.")
    return rendered


def validate_teacher_output(output: dict[str, Any]) -> None:
    required_top_level = {"score", "explanation", "resume_suggestions"}
    missing_top_level = required_top_level - set(output)
    if missing_top_level:
        raise ValueError(f"Missing required top-level fields: {sorted(missing_top_level)}")

    score = output["score"]
    if not isinstance(score, int) or not 0 <= score <= 100:
        raise ValueError("score must be an integer from 0 to 100")

    explanation = output["explanation"]
    if not isinstance(explanation, dict):
        raise ValueError("explanation must be an object")

    required_explanation = {
        "matched_qualifications",
        "missing_or_weak_qualifications",
        "overall_reasoning",
    }
    missing_explanation = required_explanation - set(explanation)
    if missing_explanation:
        raise ValueError(f"Missing explanation fields: {sorted(missing_explanation)}")

    for field in ["matched_qualifications", "missing_or_weak_qualifications"]:
        value = explanation[field]
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError(f"{field} must be an array of strings")

    if not isinstance(explanation["overall_reasoning"], str):
        raise ValueError("overall_reasoning must be a string")

    suggestions = output["resume_suggestions"]
    if not isinstance(suggestions, list):
        raise ValueError("resume_suggestions must be an array")

    required_suggestion_fields = {
        "section",
        "action",
        "suggestion",
        "evidence_from_resume",
        "job_requirement_addressed",
    }
    for index, suggestion in enumerate(suggestions):
        if not isinstance(suggestion, dict):
            raise ValueError(f"resume_suggestions[{index}] must be an object")
        missing_suggestion_fields = required_suggestion_fields - set(suggestion)
        if missing_suggestion_fields:
            raise ValueError(
                f"resume_suggestions[{index}] is missing fields: {sorted(missing_suggestion_fields)}"
            )
        for field in required_suggestion_fields:
            if not isinstance(suggestion[field], str):
                raise ValueError(f"resume_suggestions[{index}].{field} must be a string")


def load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


PAIR_COLUMNS = [
    "pair_id",
    "resume_id",
    "job_id",
    "pairing_strategy",
    "similarity_score",
    "resume_text",
    "job_description",
]


def load_pairs(processed_dir: Path, split: str, limit: int) -> list[dict[str, Any]]:
    path = processed_dir / f"teacher_pairs_{split}.csv"
    pair_rows = pd.read_csv(path)
    missing = [column for column in PAIR_COLUMNS if column not in pair_rows.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")

    count = min(limit, len(pair_rows))
    pairs = []
    for index in range(count):
        pair = pair_rows.iloc[index]
        pairs.append(
            {
                "pair_id": str(pair["pair_id"]),
                "resume_id": str(pair["resume_id"]),
                "job_id": str(pair["job_id"]),
                "pairing_strategy": str(pair["pairing_strategy"]),
                "similarity_score": float(pair["similarity_score"]),
                "resume_text": str(pair["resume_text"]),
                "job_description": str(pair["job_description"]),
            }
        )
    return pairs


def call_deepseek_raw(prompt: str, model: str) -> str:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set. Add it to .env or your shell environment.")

    request_body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "temperature": 0,
        "stream": False,
        "extra_body": {"thinking": {"type": "disabled"}},
    }
    request = Request(
        f"{DEEPSEEK_BASE_URL}/chat/completions",
        data=json.dumps(request_body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=120) as response:
            response_body = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        error_body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"DeepSeek API request failed with HTTP {error.code}: {error_body}") from error
    except URLError as error:
        raise RuntimeError(f"DeepSeek API request failed: {error.reason}") from error

    content = response_body["choices"][0]["message"]["content"]
    return content


def call_deepseek(prompt: str, model: str) -> dict[str, Any]:
    return json.loads(call_deepseek_raw(prompt, model))


def generate_teacher_output(prompt: str, model: str, max_attempts: int = 2) -> dict[str, Any]:
    last_error: Exception | None = None
    current_prompt = prompt

    for attempt in range(1, max_attempts + 1):
        try:
            raw_output = call_deepseek_raw(current_prompt, model)
            teacher_output = json.loads(raw_output) if isinstance(raw_output, str) else raw_output
            validate_teacher_output(teacher_output)
            return teacher_output
        except (json.JSONDecodeError, ValueError) as error:
            last_error = error
            if attempt == max_attempts:
                break
            if isinstance(error, json.JSONDecodeError):
                validation_message = f"JSON parsing error: {error}"
            else:
                validation_message = f"Validation error: {error}"
            current_prompt = (
                f"{prompt}\n\n"
                "Your previous response was invalid and failed validation.\n"
                f"{validation_message}\n"
                "Return the complete JSON object again using exactly these top-level keys: "
                '"score", "explanation", and "resume_suggestions".'
            )

    raise RuntimeError(f"Teacher output failed validation after {max_attempts} attempts: {last_error}")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")
        file.flush()


def load_completed_pair_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()

    completed = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path} contains invalid JSON on line {line_number}: {error}") from error
        pair_id = row.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id:
            raise ValueError(f"{path} line {line_number} is missing a valid pair_id")
        completed.add(pair_id)
    return completed


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tiny DeepSeek teacher-output pilot.")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--prompt-path", type=Path, default=Path("prompts/teacher_gold_output_prompt_v3.md"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/teacher_pilot"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--dry-run", action="store_true", help="Render prompts without calling DeepSeek.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.limit < 1:
        raise ValueError("--limit must be at least 1")

    load_env_file(args.env_file)
    template = args.prompt_path.read_text(encoding="utf-8")
    pairs = load_pairs(args.processed_dir, args.split, args.limit)

    suffix = "dry_run" if args.dry_run else "outputs"
    output_path = args.output_dir / f"deepseek_teacher_pilot_{suffix}.jsonl"
    completed_pair_ids = load_completed_pair_ids(output_path)
    written_count = 0
    skipped_count = 0

    for pair in pairs:
        if pair["pair_id"] in completed_pair_ids:
            skipped_count += 1
            continue

        prompt = render_prompt(template, pair["resume_text"], pair["job_description"])
        row = {
            "pair_id": pair["pair_id"],
            "resume_id": pair["resume_id"],
            "job_id": pair["job_id"],
            "pairing_strategy": pair["pairing_strategy"],
            "similarity_score": pair["similarity_score"],
            "prompt_version": PROMPT_VERSION,
            "model": args.model,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        if args.dry_run:
            row["rendered_prompt_chars"] = len(prompt)
        else:
            teacher_output = generate_teacher_output(prompt, args.model)
            row["teacher_output"] = teacher_output

        append_jsonl_row(output_path, row)
        completed_pair_ids.add(pair["pair_id"])
        written_count += 1

    print(f"requested_pairs: {len(pairs)}")
    print(f"skipped_existing: {skipped_count}")
    print(f"written: {written_count}")
    print(f"split: {args.split}")
    print(f"model: {args.model}")
    print(f"dry_run: {args.dry_run}")
    print(f"wrote: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
