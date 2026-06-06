import json

from src.analyze_errors import (
    categorize_model_output,
    infer_failure_sources,
    sample_weak_score_examples,
    write_phase8_report,
)


def test_sample_weak_score_examples_picks_largest_absolute_errors_per_system():
    rows = [
        {"pair_id": "a", "teacher_score": 80, "bm25": 20, "zero_shot": 75},
        {"pair_id": "b", "teacher_score": 10, "bm25": 30, "zero_shot": 90},
        {"pair_id": "c", "teacher_score": 50, "bm25": 55, "zero_shot": 60},
    ]

    samples = sample_weak_score_examples(rows, ["bm25", "zero_shot"], limit=1)

    assert samples == [
        {
            "system": "bm25",
            "pair_id": "a",
            "strategy": "",
            "teacher_score": 80.0,
            "pred_score": 20.0,
            "absolute_error": 60.0,
        },
        {
            "system": "zero_shot",
            "pair_id": "b",
            "strategy": "",
            "teacher_score": 10.0,
            "pred_score": 90.0,
            "absolute_error": 80.0,
        },
    ]


def test_sample_weak_score_examples_normalizes_raw_bm25_scale_before_ranking():
    rows = [
        {"pair_id": "a", "teacher_score": 90, "bm25": 1000},
        {"pair_id": "b", "teacher_score": 10, "bm25": 500},
    ]

    samples = sample_weak_score_examples(rows, ["bm25"], limit=1)

    assert samples == [
        {
            "system": "bm25",
            "pair_id": "b",
            "strategy": "",
            "teacher_score": 10.0,
            "pred_score": 50.0,
            "absolute_error": 40.0,
        }
    ]


def test_categorize_model_output_flags_parse_and_quality_failures():
    invalid = categorize_model_output(
        {
            "pair_id": "bad_json",
            "raw_output": "{\"score\": 40",
            "parse_error": "unterminated string",
            "teacher_score": 40,
        }
    )
    generic = categorize_model_output(
        {
            "pair_id": "generic",
            "parse_success": True,
            "teacher_score": 90,
            "parsed_output": {
                "score": 20,
                "explanation": "The candidate may be a good fit for the role.",
                "resume_suggestions": [
                    {"suggestion": "Add Kubernetes and AWS if true.", "evidence_from_resume": ""}
                ],
            },
        }
    )

    assert "Invalid JSON" in invalid["failure_modes"]
    assert "Wrong or poorly calibrated score" in generic["failure_modes"]
    assert "Overly generic explanation" in generic["failure_modes"]
    assert "Missing important job requirements" in generic["failure_modes"]
    assert "Fabricated resume content risk" in generic["failure_modes"]


def test_infer_failure_sources_maps_modes_to_likely_causes():
    sources = infer_failure_sources(
        [
            {"failure_modes": "Invalid JSON; Prompt injection vulnerability"},
            {"failure_modes": "Rewrite that removes important candidate evidence"},
        ]
    )

    assert sources["Prompt design"] == 2
    assert sources["Model limitations"] == 1
    assert sources["Compute-constrained truncation"] == 1


def test_write_phase8_report_creates_markdown_and_csv_outputs(tmp_path):
    all_methods_path = tmp_path / "all_methods.jsonl"
    model_path = tmp_path / "model.jsonl"
    output_dir = tmp_path / "phase8"
    all_methods_path.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"pair_id": "a", "strategy": "strong", "teacher_score": 80, "bm25": 20},
                {"pair_id": "b", "strategy": "weak", "teacher_score": 10, "bm25": 30},
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    model_path.write_text(
        json.dumps(
            {
                "pair_id": "a",
                "parse_success": False,
                "parse_error": "bad json",
                "teacher_score": 80,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    paths = write_phase8_report(
        all_methods_path=all_methods_path,
        model_paths={"finetuned": model_path},
        output_dir=output_dir,
        score_sample_limit=1,
        qualitative_sample_limit=5,
    )

    assert paths["score_samples"].exists()
    assert paths["qualitative_samples"].exists()
    markdown = paths["summary"].read_text(encoding="utf-8")
    assert "# Phase 8 Error Analysis Summary" in markdown
    assert "Invalid JSON" in markdown
