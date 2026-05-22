import pytest
import pandas as pd

from src.run_teacher_pilot_gemini import load_pairs, render_prompt, validate_teacher_output


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
        "rewritten_resume": "Python developer with ETL experience.",
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
        "rewritten_resume": "Resume text.",
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
