import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_MODEL = "fitmyresume"
DEFAULT_BASE_URL = "http://localhost:8000/v1"


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    instruction = str(row.get("instruction", "")).strip()
    input_text = str(row.get("input", "")).strip()
    if not instruction:
        raise ValueError("instruction-tuning row is missing instruction")
    if not input_text:
        raise ValueError("instruction-tuning row is missing input")
    return [
        {"role": "system", "content": instruction},
        {"role": "user", "content": input_text},
    ]


def strip_markdown_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return stripped


def extract_json_object(text: str) -> dict[str, Any]:
    candidates = [strip_markdown_fence(text)]
    stripped = text.strip()
    first_brace = stripped.find("{")
    last_brace = stripped.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        candidates.append(stripped[first_brace : last_brace + 1])

    errors: list[str] = []
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as error:
            errors.append(str(error))
            continue
        if isinstance(parsed, dict):
            return parsed
        errors.append("parsed JSON is not an object")

    detail = errors[-1] if errors else "empty response"
    raise ValueError(f"No valid JSON object found in model response: {detail}")


def call_vllm_chat(
    *,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    api_key: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout_seconds: float,
) -> str:
    url = f"{base_url.rstrip('/')}/chat/completions"
    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            response_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        error_body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"vLLM HTTP {error.code}: {error_body}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Could not reach vLLM server at {url}: {error}") from error

    try:
        response_json = json.loads(response_body)
        choice = response_json["choices"][0]
        if "message" in choice:
            return str(choice["message"]["content"])
        return str(choice["text"])
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Unexpected vLLM response shape: {response_body}") from error


def parse_teacher_score(row: dict[str, Any]) -> int | float | None:
    output = row.get("output")
    if not isinstance(output, str) or not output.strip():
        return None
    try:
        parsed = json.loads(output)
    except json.JSONDecodeError:
        return None
    score = parsed.get("score") if isinstance(parsed, dict) else None
    return score if isinstance(score, int | float) else None


def build_output_row(
    *,
    corpus_row: dict[str, Any],
    response_text: str,
    model: str,
    base_model: str,
    adapter_name: str,
    adapter_path: str,
    generation_config: dict[str, Any],
) -> dict[str, Any]:
    metadata = corpus_row.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    try:
        parsed_output = extract_json_object(response_text)
        parse_success = True
        parse_error = ""
    except ValueError as error:
        parsed_output = None
        parse_success = False
        parse_error = str(error)

    return {
        "pair_id": str(metadata.get("pair_id", "")),
        "split": str(metadata.get("split", "")),
        "resume_id": str(metadata.get("resume_id", "")),
        "job_id": str(metadata.get("job_id", "")),
        "pairing_strategy": str(metadata.get("pairing_strategy", "")),
        "teacher_score": parse_teacher_score(corpus_row),
        "prompt_version": str(metadata.get("prompt_version", "")),
        "model": model,
        "base_model": base_model,
        "adapter_name": adapter_name,
        "adapter_path": adapter_path,
        "generation_config": generation_config,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "raw_response": response_text,
        "parse_success": parse_success,
        "parse_error": parse_error,
        "parsed_output": parsed_output,
    }


def run_inference(
    *,
    input_path: Path,
    output_path: Path,
    model: str,
    base_url: str,
    api_key: str,
    base_model: str,
    adapter_name: str,
    adapter_path: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout_seconds: float,
    limit: int | None,
    sleep_seconds: float,
) -> dict[str, int]:
    rows = read_jsonl(input_path)
    if limit is not None:
        rows = rows[:limit]

    generation_config = {
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    output_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        messages = build_messages(row)
        response_text = call_vllm_chat(
            base_url=base_url,
            model=model,
            messages=messages,
            api_key=api_key,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
        )
        output_rows.append(
            build_output_row(
                corpus_row=row,
                response_text=response_text,
                model=model,
                base_model=base_model,
                adapter_name=adapter_name,
                adapter_path=adapter_path,
                generation_config=generation_config,
            )
        )
        print(f"processed {index}/{len(rows)}: {output_rows[-1]['pair_id']}")
        if sleep_seconds > 0 and index < len(rows):
            time.sleep(sleep_seconds)

    write_jsonl(output_path, output_rows)
    parse_success = sum(1 for row in output_rows if row["parse_success"])
    return {
        "rows": len(output_rows),
        "parse_success": parse_success,
        "parse_failed": len(output_rows) - parse_success,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fine-tuned Qwen LoRA inference through a vLLM OpenAI-compatible endpoint."
    )
    parser.add_argument("--input", type=Path, required=True, help="Instruction-tuning JSONL input.")
    parser.add_argument("--output", type=Path, required=True, help="JSONL path for model outputs.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="vLLM model or LoRA name to request.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible vLLM base URL.")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY", ""),
        help="API key if vLLM was started with --api-key. Defaults to OPENAI_API_KEY or empty.",
    )
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter-name", default=DEFAULT_MODEL)
    parser.add_argument("--adapter-path", default="")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for smoke tests.")
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="Optional delay between requests.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    summary = run_inference(
        input_path=args.input,
        output_path=args.output,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        base_model=args.base_model,
        adapter_name=args.adapter_name,
        adapter_path=args.adapter_path,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        timeout_seconds=args.timeout_seconds,
        limit=args.limit,
        sleep_seconds=args.sleep_seconds,
    )
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
