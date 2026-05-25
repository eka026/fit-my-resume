"""
Build SFT dataset from teacher pilot outputs.

Reads:
- results/teacher_pilot/deepseek_teacher_pilot_outputs.jsonl  (teacher outputs)
- data/processed/resumes_train.csv, resumes_validation.csv    (resume_id -> resume_text)
- data/processed/jobs_train.csv, jobs_validation.csv          (job_id -> job_description)
- prompts/teacher_gold_output_prompt_v3.md                    (for instruction)

Writes:
- data/processed/sft_train.jsonl
- data/processed/sft_validation.jsonl
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd


def build_instruction(template_path: Path) -> str:
    """Strip placeholders out of the prompt template to get the static instruction."""
    template = template_path.read_text(encoding="utf-8")
    return template.replace(
        "RESUME:\n{{resume_text}}\n\nJOB_DESCRIPTION:\n{{job_description}}",
        "The resume and job description are provided in the input.",
    )


def build_input(resume_text: str, job_description: str) -> str:
    return f"RESUME:\n{resume_text}\n\nJOB_DESCRIPTION:\n{job_description}"


def load_text_map(csv_path: Path, id_col: str, text_col: str) -> dict:
    """Load a CSV and return {id: text}."""
    if not csv_path.exists():
        print(f"WARNING: {csv_path} does not exist, skipping")
        return {}
    df = pd.read_csv(csv_path)
    if id_col not in df.columns or text_col not in df.columns:
        print(f"WARNING: {csv_path} missing {id_col} or {text_col}, has: {df.columns.tolist()}")
        return {}
    return {str(row[id_col]): str(row[text_col]) for _, row in df.iterrows()}


def main(args):
    instruction = build_instruction(args.prompt_path)

    train_resumes = load_text_map(args.processed_dir / "resumes_train.csv", "resume_id", "resume_text")
    val_resumes = load_text_map(args.processed_dir / "resumes_validation.csv", "resume_id", "resume_text")

    train_jobs = load_text_map(args.processed_dir / "jobs_train.csv", "job_id", "job_description")
    val_jobs = load_text_map(args.processed_dir / "jobs_validation.csv", "job_id", "job_description")

    all_resumes = {**val_resumes, **train_resumes}
    all_jobs = {**val_jobs, **train_jobs}

    print(f"loaded {len(train_resumes)} train resumes, {len(val_resumes)} val resumes")
    print(f"loaded {len(train_jobs)} train jobs, {len(val_jobs)} val jobs")

    train_out = []
    val_out = []
    skipped = {
        "missing_resume": 0,
        "missing_job": 0,
        "bad_teacher": 0,
        "too_long": 0,
        "unknown_split": 0,
    }

    with args.teacher_outputs.open() as f:
        for line in f:
            row = json.loads(line)
            pair_id = row["pair_id"]
            resume_id = str(row.get("resume_id"))
            job_id = str(row.get("job_id"))
            strategy = row.get("pairing_strategy", "unknown")
            teacher_output = row.get("teacher_output")

            if not teacher_output:
                skipped["bad_teacher"] += 1
                continue

            resume_text = all_resumes.get(resume_id)
            job_description = all_jobs.get(job_id)

            if not resume_text:
                skipped["missing_resume"] += 1
                continue
            if not job_description:
                skipped["missing_job"] += 1
                continue

            if resume_id in val_resumes:
                split = "validation"
            elif resume_id in train_resumes:
                split = "train"
            elif pair_id.startswith("train_"):
                split = "train"
            elif pair_id.startswith("val") or pair_id.startswith("validation"):
                split = "validation"
            else:
                skipped["unknown_split"] += 1
                continue

            input_text = build_input(resume_text, job_description)
            output_text = json.dumps(teacher_output, ensure_ascii=False)

            total_chars = len(instruction) + len(input_text) + len(output_text)
            if total_chars > 28000:
                skipped["too_long"] += 1
                continue

            sft_row = {
                "pair_id": pair_id,
                "strategy": strategy,
                "instruction": instruction,
                "input": input_text,
                "output": output_text,
            }

            if split == "train":
                train_out.append(sft_row)
            else:
                val_out.append(sft_row)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with (args.output_dir / "sft_train.jsonl").open("w") as f:
        for row in train_out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with (args.output_dir / "sft_validation.jsonl").open("w") as f:
        for row in val_out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print()
    print(f"train rows written: {len(train_out)}")
    print(f"validation rows written: {len(val_out)}")
    print(f"skipped: {skipped}")
    print(f"wrote: {args.output_dir / 'sft_train.jsonl'}")
    print(f"wrote: {args.output_dir / 'sft_validation.jsonl'}")

    if train_out:
        print(f"train strategy distribution: {dict(Counter(r['strategy'] for r in train_out))}")
    if val_out:
        print(f"validation strategy distribution: {dict(Counter(r['strategy'] for r in val_out))}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--teacher-outputs",
        type=Path,
        default=Path("results/teacher_pilot/deepseek_teacher_pilot_outputs.jsonl"),
    )
    p.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    p.add_argument(
        "--prompt-path",
        type=Path,
        default=Path("prompts/teacher_gold_output_prompt_v3.md"),
    )
    p.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    args = p.parse_args()
    main(args)