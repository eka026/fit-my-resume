import json
import subprocess
import sys

import pandas as pd
import pytest

from src.build_instruction_corpus import (
    build_corpus,
    build_instruction_row,
    validate_strict_teacher_output,
)


def expected_teacher_output():
    return {
        "score": 42,
        "explanation": {
            "matched_qualifications": ["Resume evidence matches one job requirement."],
            "missing_or_weak_qualifications": ["One important job requirement is missing."],
            "overall_reasoning": "The resume has some overlap but also important gaps.",
        },
        "resume_suggestions": [
            {
                "section": "experience",
                "action": "emphasize",
                "suggestion": "Emphasize the supported project work.",
                "evidence_from_resume": "Led a relevant project.",
                "job_requirement_addressed": "Project leadership.",
            }
        ],
    }


def test_build_instruction_row_contains_training_text_and_metadata():
    pair = {
        "pair_id": "train_resume_1_job_1_strong_hybrid",
        "split": "train",
        "resume_id": "resume_1",
        "job_id": "job_1",
        "pairing_strategy": "strong_hybrid",
        "similarity_score": 0.75,
        "resume_text": "Led a relevant project.",
        "job_description": "Needs project leadership.",
    }
    teacher_row = {
        "pair_id": "train_resume_1_job_1_strong_hybrid",
        "prompt_version": "teacher_gold_output_prompt_v3",
        "model": "deepseek-v4-flash",
        "teacher_output": expected_teacher_output(),
    }

    row = build_instruction_row(pair, teacher_row)

    assert row["instruction"].startswith("Evaluate the resume against the job description")
    assert row["input"] == (
        "RESUME:\nLed a relevant project.\n\n"
        "JOB_DESCRIPTION:\nNeeds project leadership."
    )
    assert json.loads(row["output"]) == expected_teacher_output()
    assert row["metadata"] == {
        "pair_id": "train_resume_1_job_1_strong_hybrid",
        "split": "train",
        "resume_id": "resume_1",
        "job_id": "job_1",
        "pairing_strategy": "strong_hybrid",
        "similarity_score": 0.75,
        "prompt_version": "teacher_gold_output_prompt_v3",
        "model": "deepseek-v4-flash",
    }


def test_validate_strict_teacher_output_rejects_invalid_section():
    output = expected_teacher_output()
    output["resume_suggestions"][0]["section"] = "skillssection"

    with pytest.raises(ValueError, match="section"):
        validate_strict_teacher_output(output)


def test_validate_strict_teacher_output_rejects_invalid_action():
    output = expected_teacher_output()
    output["resume_suggestions"][0]["action"] = "rewrite"

    with pytest.raises(ValueError, match="action"):
        validate_strict_teacher_output(output)


def test_build_corpus_writes_joined_jsonl_and_summary(tmp_path):
    pairs_path = tmp_path / "teacher_pairs_train.csv"
    outputs_path = tmp_path / "deepseek_outputs.jsonl"
    corpus_path = tmp_path / "instruction_tuning_train.jsonl"

    pd.DataFrame(
        [
            {
                "pair_id": "train_resume_1_job_1_strong_hybrid",
                "split": "train",
                "resume_id": "resume_1",
                "job_id": "job_1",
                "pairing_strategy": "strong_hybrid",
                "similarity_score": 0.75,
                "resume_category": "IT",
                "job_position_title": "Project Lead",
                "resume_text": "Led a relevant project.",
                "job_description": "Needs project leadership.",
            }
        ]
    ).to_csv(pairs_path, index=False)
    outputs_path.write_text(
        json.dumps(
            {
                "pair_id": "train_resume_1_job_1_strong_hybrid",
                "prompt_version": "teacher_gold_output_prompt_v3",
                "model": "deepseek-v4-flash",
                "teacher_output": expected_teacher_output(),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = build_corpus(pairs_path, outputs_path, corpus_path)

    assert summary == {
        "pairs_read": 1,
        "teacher_outputs_read": 1,
        "written": 1,
        "missing": 0,
        "invalid": 0,
    }
    rows = [json.loads(line) for line in corpus_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["metadata"]["pair_id"] == "train_resume_1_job_1_strong_hybrid"
    assert json.loads(rows[0]["output"]) == expected_teacher_output()


def test_build_corpus_can_skip_invalid_teacher_rows(tmp_path):
    pairs_path = tmp_path / "teacher_pairs_train.csv"
    outputs_path = tmp_path / "deepseek_outputs.jsonl"
    corpus_path = tmp_path / "instruction_tuning_train.jsonl"
    valid_output = expected_teacher_output()
    invalid_output = expected_teacher_output()
    invalid_output["resume_suggestions"][0]["action"] = "rewrite"

    pd.DataFrame(
        [
            {
                "pair_id": "pair_valid",
                "split": "train",
                "resume_id": "resume_1",
                "job_id": "job_1",
                "pairing_strategy": "strong_hybrid",
                "similarity_score": 0.75,
                "resume_category": "IT",
                "job_position_title": "Project Lead",
                "resume_text": "Led a relevant project.",
                "job_description": "Needs project leadership.",
            },
            {
                "pair_id": "pair_invalid",
                "split": "train",
                "resume_id": "resume_2",
                "job_id": "job_2",
                "pairing_strategy": "weak_random",
                "similarity_score": 0.05,
                "resume_category": "IT",
                "job_position_title": "Designer",
                "resume_text": "Managed servers.",
                "job_description": "Needs visual design.",
            },
        ]
    ).to_csv(pairs_path, index=False)
    outputs_path.write_text(
        "\n".join(
            [
                json.dumps({"pair_id": "pair_valid", "teacher_output": valid_output}),
                json.dumps({"pair_id": "pair_invalid", "teacher_output": invalid_output}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = build_corpus(pairs_path, outputs_path, corpus_path, skip_invalid=True)

    assert summary["written"] == 1
    assert summary["invalid"] == 1
    rows = [json.loads(line) for line in corpus_path.read_text(encoding="utf-8").splitlines()]
    assert [row["metadata"]["pair_id"] for row in rows] == ["pair_valid"]


def test_cli_runs_when_called_as_script_path(tmp_path):
    pairs_path = tmp_path / "teacher_pairs_train.csv"
    outputs_path = tmp_path / "deepseek_outputs.jsonl"
    corpus_path = tmp_path / "instruction_tuning_train.jsonl"

    pd.DataFrame(
        [
            {
                "pair_id": "pair_valid",
                "split": "train",
                "resume_id": "resume_1",
                "job_id": "job_1",
                "pairing_strategy": "strong_hybrid",
                "similarity_score": 0.75,
                "resume_category": "IT",
                "job_position_title": "Project Lead",
                "resume_text": "Led a relevant project.",
                "job_description": "Needs project leadership.",
            }
        ]
    ).to_csv(pairs_path, index=False)
    outputs_path.write_text(
        json.dumps({"pair_id": "pair_valid", "teacher_output": expected_teacher_output()}) + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "src/build_instruction_corpus.py",
            "--pairs-path",
            str(pairs_path),
            "--teacher-outputs-path",
            str(outputs_path),
            "--output-path",
            str(corpus_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "written: 1" in result.stdout
    assert corpus_path.exists()
