import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PROMPT_VERSION = "teacher_gold_output_prompt_v1"
DEFAULT_MODEL = "gemini-2.5-flash"

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
        "rewritten_resume": {"type": "string"},
    },
    "required": ["score", "explanation", "rewritten_resume"],
}


def render_prompt(template: str, resume_text: str, job_description: str) -> str:
    rendered = template.replace("{{resume_text}}", resume_text)
    rendered = rendered.replace("{{job_description}}", job_description)
    if "{{resume_text}}" in rendered or "{{job_description}}" in rendered:
        raise ValueError("Prompt still contains unfilled placeholders.")
    return rendered


def validate_teacher_output(output: dict[str, Any]) -> None:
    required_top_level = {"score", "explanation", "rewritten_resume"}
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

    if not isinstance(output["rewritten_resume"], str):
        raise ValueError("rewritten_resume must be a string")


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


def call_gemini(prompt: str, model: str) -> dict[str, Any]:
    try:
        from google import genai
        from google.genai import types
    except ImportError as error:
        raise RuntimeError(
            "Missing Gemini SDK. Install it with: python -m pip install google-genai"
        ) from error

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Add it to .env or your shell environment.")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=TEACHER_SCHEMA,
            temperature=0.2,
        ),
    )
    return json.loads(response.text)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tiny Gemini teacher-output pilot.")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--prompt-path", type=Path, default=Path("prompts/teacher_gold_output_prompt_v1.md"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/teacher_pilot"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--dry-run", action="store_true", help="Render prompts without calling Gemini.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.limit < 1:
        raise ValueError("--limit must be at least 1")

    load_env_file(args.env_file)
    template = args.prompt_path.read_text(encoding="utf-8")
    pairs = load_pairs(args.processed_dir, args.split, args.limit)

    rows = []
    for pair in pairs:
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
            teacher_output = call_gemini(prompt, args.model)
            validate_teacher_output(teacher_output)
            row["teacher_output"] = teacher_output

        rows.append(row)

    suffix = "dry_run" if args.dry_run else "outputs"
    output_path = args.output_dir / f"gemini_teacher_pilot_{suffix}.jsonl"
    write_jsonl(output_path, rows)

    print(f"pairs: {len(rows)}")
    print(f"split: {args.split}")
    print(f"model: {args.model}")
    print(f"dry_run: {args.dry_run}")
    print(f"wrote: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
