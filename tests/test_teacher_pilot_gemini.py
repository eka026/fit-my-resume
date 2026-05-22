import pytest

from src.run_teacher_pilot_gemini import render_prompt, validate_teacher_output


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
