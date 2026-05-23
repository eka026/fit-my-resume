import json
from pathlib import Path

import pytest
import pandas as pd

from src.run_teacher_pilot import (
    DEFAULT_MODEL,
    PROMPT_VERSION,
    append_jsonl_row,
    call_deepseek,
    generate_teacher_output,
    load_completed_pair_ids,
    load_pairs,
    parse_args,
    render_prompt,
    validate_teacher_output,
)


def test_render_prompt_fills_resume_and_job_placeholders():
    template = "RESUME:\n{{resume_text}}\nJOB:\n{{job_description}}"

    rendered = render_prompt(
        template=template,
        resume_text="Built Python ETL pipelines.",
        job_description="Needs Python data engineering.",
    )

    assert "{{resume_text}}" not in rendered
    assert "{{job_description}}" not in rendered
    assert "Built Python ETL pipelines." in rendered
    assert "Needs Python data engineering." in rendered


def test_validate_teacher_output_accepts_expected_schema():
    output = {
        "score": 75,
        "explanation": {
            "matched_qualifications": ["Python experience matches the role."],
            "missing_or_weak_qualifications": ["Cloud experience is not shown."],
            "overall_reasoning": "The resume has relevant programming experience but lacks cloud evidence.",
        },
        "resume_suggestions": [
            {
                "section": "skills",
                "action": "emphasize",
                "suggestion": "Move Python and ETL experience closer to the top of the skills section.",
                "evidence_from_resume": "Python developer with ETL experience.",
                "job_requirement_addressed": "Python data engineering experience.",
            }
        ],
    }

    validate_teacher_output(output)


def test_validate_teacher_output_rejects_score_outside_range():
    output = {
        "score": 101,
        "explanation": {
            "matched_qualifications": [],
            "missing_or_weak_qualifications": [],
            "overall_reasoning": "Too high.",
        },
        "resume_suggestions": [],
    }

    with pytest.raises(ValueError, match="score"):
        validate_teacher_output(output)


def test_load_pairs_reads_teacher_pair_file(tmp_path):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    pd.DataFrame(
        [
            {
                "pair_id": "train_resume_1_job_2_strong_hybrid",
                "split": "train",
                "resume_id": "resume_1",
                "job_id": "job_2",
                "pairing_strategy": "strong_hybrid",
                "similarity_score": 0.75,
                "resume_category": "Data Science",
                "job_position_title": "ML Engineer",
                "resume_text": "Python and machine learning resume.",
                "job_description": "Machine learning job description.",
            },
            {
                "pair_id": "train_resume_3_job_4_weak_random",
                "split": "train",
                "resume_id": "resume_3",
                "job_id": "job_4",
                "pairing_strategy": "weak_random",
                "similarity_score": 0.02,
                "resume_category": "HR",
                "job_position_title": "Warehouse Associate",
                "resume_text": "Recruiting and onboarding resume.",
                "job_description": "Warehouse inventory job description.",
            },
        ]
    ).to_csv(processed_dir / "teacher_pairs_train.csv", index=False)

    pairs = load_pairs(processed_dir=processed_dir, split="train", limit=1)

    assert pairs == [
        {
            "pair_id": "train_resume_1_job_2_strong_hybrid",
            "resume_id": "resume_1",
            "job_id": "job_2",
            "pairing_strategy": "strong_hybrid",
            "similarity_score": 0.75,
            "resume_text": "Python and machine learning resume.",
            "job_description": "Machine learning job description.",
        }
    ]


def test_load_completed_pair_ids_reads_existing_jsonl(tmp_path):
    output_path = tmp_path / "teacher_outputs.jsonl"
    output_path.write_text(
        "\n".join(
            [
                json.dumps({"pair_id": "pair_1", "teacher_output": {}}),
                json.dumps({"pair_id": "pair_2", "teacher_output": {}}),
                "",
            ]
        ),
        encoding="utf-8",
    )

    assert load_completed_pair_ids(output_path) == {"pair_1", "pair_2"}


def test_append_jsonl_row_writes_one_checkpoint_row(tmp_path):
    output_path = tmp_path / "nested" / "teacher_outputs.jsonl"

    append_jsonl_row(output_path, {"pair_id": "pair_1", "score": 10})
    append_jsonl_row(output_path, {"pair_id": "pair_2", "score": 20})

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["pair_id"] for line in lines] == ["pair_1", "pair_2"]


def test_main_skips_completed_pairs_and_appends_new_rows(tmp_path, monkeypatch):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    pd.DataFrame(
        [
            {
                "pair_id": "pair_1",
                "resume_id": "resume_1",
                "job_id": "job_1",
                "pairing_strategy": "strong_hybrid",
                "similarity_score": 0.9,
                "resume_text": "Python resume.",
                "job_description": "Python job.",
            },
            {
                "pair_id": "pair_2",
                "resume_id": "resume_2",
                "job_id": "job_2",
                "pairing_strategy": "weak_random",
                "similarity_score": 0.1,
                "resume_text": "Design resume.",
                "job_description": "Accounting job.",
            },
        ]
    ).to_csv(processed_dir / "teacher_pairs_train.csv", index=False)
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("RESUME {{resume_text}} JOB {{job_description}}", encoding="utf-8")
    output_dir = tmp_path / "outputs"
    output_path = output_dir / "deepseek_teacher_pilot_outputs.jsonl"
    append_jsonl_row(output_path, {"pair_id": "pair_1", "teacher_output": {}})

    calls = []

    def fake_generate_teacher_output(prompt, model):
        calls.append((prompt, model))
        return {
            "score": 20,
            "explanation": {
                "matched_qualifications": [],
                "missing_or_weak_qualifications": [],
                "overall_reasoning": "Weak fit.",
            },
            "resume_suggestions": [],
        }

    monkeypatch.setattr("src.run_teacher_pilot.generate_teacher_output", fake_generate_teacher_output)

    from src.run_teacher_pilot import main

    main(
        [
            "--processed-dir",
            str(processed_dir),
            "--prompt-path",
            str(prompt_path),
            "--output-dir",
            str(output_dir),
            "--split",
            "train",
            "--limit",
            "2",
        ]
    )

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["pair_id"] for line in lines] == ["pair_1", "pair_2"]
    assert len(calls) == 1
    assert "Design resume." in calls[0][0]


def test_default_prompt_uses_v2():
    args = parse_args([])

    assert PROMPT_VERSION == "teacher_gold_output_prompt_v3"
    assert args.prompt_path.as_posix() == "prompts/teacher_gold_output_prompt_v3.md"


def test_v3_prompt_forbids_unsupported_target_role_repositioning():
    prompt = Path("prompts/teacher_gold_output_prompt_v3.md").read_text(encoding="utf-8")

    assert "Do not suggest repositioning the resume toward a target role" in prompt
    assert "direct support for that role, domain, or function" in prompt


def test_call_deepseek_uses_flash_model_json_mode(monkeypatch):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return None

        def read(self):
            return (
                b'{"choices": [{"message": {"content": "{\\"score\\": 80, '
                b'\\"explanation\\": {\\"matched_qualifications\\": [], '
                b'\\"missing_or_weak_qualifications\\": [], '
                b'\\"overall_reasoning\\": \\"Good fit.\\"}, '
                b'\\"resume_suggestions\\": []}"}}]}'
            )

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr("src.run_teacher_pilot.urlopen", fake_urlopen)

    output = call_deepseek("Return JSON.", DEFAULT_MODEL)

    assert output["score"] == 80
    assert captured["url"] == "https://api.deepseek.com/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["body"]["model"] == "deepseek-v4-flash"
    assert captured["body"]["response_format"] == {"type": "json_object"}
    assert captured["body"]["extra_body"] == {"thinking": {"type": "disabled"}}


def test_generate_teacher_output_retries_after_invalid_schema(monkeypatch):
    responses = [
        {"score": 80, "explanation": {"matched_qualifications": [], "missing_or_weak_qualifications": [], "overall_reasoning": "Good fit."}},
        {
            "score": 80,
            "explanation": {
                "matched_qualifications": [],
                "missing_or_weak_qualifications": [],
                "overall_reasoning": "Good fit.",
            },
            "resume_suggestions": [],
        },
    ]
    prompts = []

    def fake_call_deepseek_raw(prompt, model):
        prompts.append(prompt)
        return responses.pop(0)

    monkeypatch.setattr("src.run_teacher_pilot.call_deepseek_raw", fake_call_deepseek_raw)

    output = generate_teacher_output("Original prompt.", DEFAULT_MODEL, max_attempts=2)

    assert output["resume_suggestions"] == []
    assert len(prompts) == 2
    assert "Missing required top-level fields" in prompts[1]
    assert '"resume_suggestions"' in prompts[1]


def test_generate_teacher_output_retries_after_invalid_json(monkeypatch):
    responses = [
        "I cannot provide JSON.",
        {
            "score": 70,
            "explanation": {
                "matched_qualifications": [],
                "missing_or_weak_qualifications": [],
                "overall_reasoning": "Partial fit.",
            },
            "resume_suggestions": [],
        },
    ]
    prompts = []

    def fake_call_deepseek_raw(prompt, model):
        prompts.append(prompt)
        return responses.pop(0)

    monkeypatch.setattr("src.run_teacher_pilot.call_deepseek_raw", fake_call_deepseek_raw)

    output = generate_teacher_output("Original prompt.", DEFAULT_MODEL, max_attempts=2)

    assert output["score"] == 70
    assert len(prompts) == 2
    assert "Your previous response was invalid" in prompts[1]
    assert "JSON parsing error" in prompts[1]
