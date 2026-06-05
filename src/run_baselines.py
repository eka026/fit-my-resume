import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from sklearn.metrics.pairwise import cosine_similarity


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9+#.]+")


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


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def split_instruction_input(text: str) -> tuple[str, str]:
    resume_marker = "RESUME:\n"
    job_marker = "\n\nJOB_DESCRIPTION:\n"
    if resume_marker not in text or job_marker not in text:
        raise ValueError("Instruction input must contain RESUME and JOB_DESCRIPTION sections")
    resume_start = text.index(resume_marker) + len(resume_marker)
    job_start = text.index(job_marker)
    resume = text[resume_start:job_start]
    job_description = text[job_start + len(job_marker) :]
    return resume.strip(), job_description.strip()


def bm25_scores(query: str, documents: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    query_terms = tokenize(query)
    document_tokens = [tokenize(document) for document in documents]
    if not query_terms or not document_tokens:
        return [0.0 for _ in documents]

    document_count = len(document_tokens)
    document_lengths = [len(tokens) for tokens in document_tokens]
    average_length = sum(document_lengths) / document_count if document_count else 0.0
    document_frequencies: Counter[str] = Counter()
    for tokens in document_tokens:
        document_frequencies.update(set(tokens))

    scores: list[float] = []
    for tokens, document_length in zip(document_tokens, document_lengths):
        term_frequencies = Counter(tokens)
        score = 0.0
        for term in query_terms:
            frequency = term_frequencies.get(term, 0)
            if frequency == 0:
                continue
            idf = math.log(1 + (document_count - document_frequencies[term] + 0.5) / (document_frequencies[term] + 0.5))
            denominator = frequency + k1 * (1 - b + b * document_length / average_length) if average_length else 1
            score += idf * (frequency * (k1 + 1) / denominator)
        scores.append(score)
    return scores


def scale_score(value: float) -> float:
    return round(max(0.0, min(100.0, value * 100)), 4)


def teacher_score(row: dict[str, Any]) -> float | None:
    output = row.get("output")
    if isinstance(output, str):
        try:
            parsed = json.loads(output)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict) and isinstance(parsed.get("score"), int | float):
            return float(parsed["score"])
    return None


def metadata_for(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def build_output_row(row: dict[str, Any], model: str, pred_score: float, raw_score: float) -> dict[str, Any]:
    metadata = metadata_for(row)
    return {
        "pair_id": metadata.get("pair_id", row.get("pair_id")),
        "split": metadata.get("split"),
        "resume_id": metadata.get("resume_id"),
        "job_id": metadata.get("job_id"),
        "pairing_strategy": metadata.get("pairing_strategy"),
        "model": model,
        "gt_score": teacher_score(row),
        "pred_score": pred_score,
        "raw_similarity": round(raw_score, 6),
    }


def generate_bm25_outputs(input_path: Path, output_path: Path) -> list[dict[str, Any]]:
    source_rows = read_jsonl(input_path)
    parsed_rows: list[tuple[dict[str, Any], str, str]] = []
    for row in source_rows:
        input_text = row.get("input")
        if not isinstance(input_text, str):
            continue
        resume, job_description = split_instruction_input(input_text)
        parsed_rows.append((row, resume, job_description))

    raw_scores = [bm25_scores(job_description, [resume])[0] for row, resume, job_description in parsed_rows]
    max_score = max(raw_scores) if raw_scores else 0.0
    output_rows: list[dict[str, Any]] = []
    for (row, _resume, _job_description), raw_score in zip(parsed_rows, raw_scores):
        normalized = raw_score / max_score if max_score else 0.0
        output_rows.append(build_output_row(row, "bm25", scale_score(normalized), raw_score))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in output_rows) + ("\n" if output_rows else ""),
        encoding="utf-8",
    )
    return output_rows


def generate_sentence_transformer_outputs(input_path: Path, output_path: Path, model_name: str) -> list[dict[str, Any]]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "sentence-transformers is not installed. Install it in Colab or run BM25 locally with --baseline bm25."
        ) from error

    source_rows = read_jsonl(input_path)
    parsed_rows: list[tuple[dict[str, Any], str, str]] = []
    for row in source_rows:
        input_text = row.get("input")
        if isinstance(input_text, str):
            resume, job_description = split_instruction_input(input_text)
            parsed_rows.append((row, resume, job_description))

    model = SentenceTransformer(model_name)
    resumes = [resume for _row, resume, _job_description in parsed_rows]
    jobs = [job_description for _row, _resume, job_description in parsed_rows]
    resume_embeddings = model.encode(resumes, normalize_embeddings=True)
    job_embeddings = model.encode(jobs, normalize_embeddings=True)
    similarities = cosine_similarity(resume_embeddings, job_embeddings).diagonal()

    output_rows = [
        build_output_row(row, "sentence_transformer", scale_score((float(score) + 1) / 2), float(score))
        for (row, _resume, _job_description), score in zip(parsed_rows, similarities)
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in output_rows) + ("\n" if output_rows else ""),
        encoding="utf-8",
    )
    return output_rows


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate FitMyResume baseline outputs.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", choices=["bm25", "sentence-transformer"], required=True)
    parser.add_argument("--sentence-model", default="sentence-transformers/all-mpnet-base-v2")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.baseline == "bm25":
        rows = generate_bm25_outputs(args.input, args.output)
    else:
        rows = generate_sentence_transformer_outputs(args.input, args.output, args.sentence_model)
    print(f"wrote {len(rows)} rows: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
