import json
from pathlib import Path

from src.run_finetuned_vllm_inference import (
    build_messages,
    build_output_row,
    extract_json_object,
    read_jsonl,
)


def test_build_messages_uses_instruction_and_input_as_system_and_user():
    row = {
        "instruction": "Return JSON only.",
        "input": "RESUME:\nBuilt APIs.\n\nJOB_DESCRIPTION:\nNeeds API work.",
    }

    messages = build_messages(row)

    assert messages == [
        {"role": "system", "content": "Return JSON only."},
        {"role": "user", "content": "RESUME:\nBuilt APIs.\n\nJOB_DESCRIPTION:\nNeeds API work."},
    ]


def test_extract_json_object_accepts_plain_json_and_fenced_json():
    expected = {"score": 80, "explanation": {}, "resume_suggestions": []}

    assert extract_json_object(json.dumps(expected)) == expected
    assert extract_json_object(f"```json\n{json.dumps(expected)}\n```") == expected


def test_extract_json_object_raises_for_non_json_text():
    try:
        extract_json_object("The score is high.")
    except ValueError as error:
        assert "No valid JSON object" in str(error)
    else:
        raise AssertionError("expected invalid text to raise ValueError")


def test_build_output_row_records_parse_success_and_metadata():
    corpus_row = {
        "instruction": "Return JSON only.",
        "input": "RESUME:\nBuilt APIs.",
        "output": "{\"score\":75}",
        "metadata": {
            "pair_id": "validation_resume_1_job_1_strong_hybrid",
            "split": "validation",
            "prompt_version": "teacher_gold_output_prompt_v3",
        },
    }
    response_text = json.dumps(
        {
            "score": 70,
            "explanation": {
                "matched_qualifications": ["API work"],
                "missing_or_weak_qualifications": [],
                "overall_reasoning": "Strong overlap.",
            },
            "resume_suggestions": [],
        }
    )

    output_row = build_output_row(
        corpus_row=corpus_row,
        response_text=response_text,
        model="fitmyresume",
        base_model="Qwen/Qwen2.5-7B-Instruct",
        adapter_name="fitmyresume",
        adapter_path="/content/drive/MyDrive/fit-my-resume/models/qwen25-7b-fitmyresume-lora-v2/final",
        generation_config={"temperature": 0.0, "max_tokens": 1024},
    )

    assert output_row["pair_id"] == "validation_resume_1_job_1_strong_hybrid"
    assert output_row["split"] == "validation"
    assert output_row["parse_success"] is True
    assert output_row["parsed_output"]["score"] == 70
    assert output_row["parse_error"] == ""
    assert output_row["model"] == "fitmyresume"
    assert output_row["base_model"] == "Qwen/Qwen2.5-7B-Instruct"
    assert output_row["adapter_name"] == "fitmyresume"
    assert output_row["prompt_version"] == "teacher_gold_output_prompt_v3"


def test_read_jsonl_skips_blank_lines(tmp_path):
    path = Path(tmp_path) / "rows.jsonl"
    path.write_text('{"a": 1}\n\n{"a": 2}\n', encoding="utf-8")

    assert read_jsonl(path) == [{"a": 1}, {"a": 2}]
