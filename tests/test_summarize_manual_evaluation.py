import csv

import pytest

from src.summarize_manual_evaluation import (
    summarize_manual_evaluation,
    write_manual_evaluation_summary,
)


FIELDNAMES = [
    "sample_label",
    "pair_id",
    "split",
    "resume_id",
    "job_id",
    "pairing_strategy",
    "resume_text",
    "job_description",
    "generated_rewrite",
    "evaluator_id",
    "relevance_1_to_5",
    "faithfulness_1_to_5",
    "clarity_usefulness_1_to_5",
    "overall_preference_1_to_5",
    "fabricated_content_flag",
    "qualitative_notes",
]


def write_form(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def completed_row(pair_id, evaluator_id, relevance, faithfulness, clarity, overall, fabricated, notes):
    return {
        "sample_label": "validation_sample",
        "pair_id": pair_id,
        "split": "validation",
        "resume_id": f"resume_{pair_id}",
        "job_id": f"job_{pair_id}",
        "pairing_strategy": "strong_hybrid",
        "resume_text": "Resume",
        "job_description": "Job",
        "generated_rewrite": "Rewrite",
        "evaluator_id": evaluator_id,
        "relevance_1_to_5": relevance,
        "faithfulness_1_to_5": faithfulness,
        "clarity_usefulness_1_to_5": clarity,
        "overall_preference_1_to_5": overall,
        "fabricated_content_flag": fabricated,
        "qualitative_notes": notes,
    }


def test_summarize_manual_evaluation_reports_averages_and_fabrication_flags(tmp_path):
    form_path = tmp_path / "manual_form.csv"
    write_form(
        form_path,
        [
            completed_row("pair_1", "a", "5", "4", "5", "4", "no", "Specific and useful."),
            completed_row("pair_1", "b", "3", "2", "4", "3", "yes", "May add unsupported CRM skill."),
            completed_row("pair_2", "a", "4", "5", "4", "5", "false", ""),
        ],
    )

    summary = summarize_manual_evaluation(form_path)

    assert summary["rated_rows"] == 3
    assert summary["unique_examples"] == 2
    assert summary["unique_evaluators"] == 2
    assert summary["fabrication_flags"] == 1
    assert summary["average_relevance_1_to_5"] == pytest.approx(4.0)
    assert summary["average_faithfulness_1_to_5"] == pytest.approx(3.6667)
    assert summary["common_qualitative_issues"] == [
        "Specific and useful.",
        "May add unsupported CRM skill.",
    ]


def test_summarize_manual_evaluation_handles_blank_unrated_forms(tmp_path):
    form_path = tmp_path / "manual_form.csv"
    write_form(
        form_path,
        [
            completed_row("pair_1", "", "", "", "", "", "", ""),
        ],
    )

    summary = summarize_manual_evaluation(form_path)

    assert summary["rated_rows"] == 0
    assert summary["unique_examples"] == 1
    assert summary["average_relevance_1_to_5"] is None
    assert summary["fabrication_flags"] == 0


def test_summarize_manual_evaluation_rejects_out_of_range_scores(tmp_path):
    form_path = tmp_path / "manual_form.csv"
    write_form(
        form_path,
        [
            completed_row("pair_1", "a", "6", "4", "4", "4", "no", ""),
        ],
    )

    with pytest.raises(ValueError, match="relevance_1_to_5"):
        summarize_manual_evaluation(form_path)


def test_write_manual_evaluation_summary_outputs_csv_and_markdown(tmp_path):
    summary = {
        "source_path": "manual_form.csv",
        "total_rows": 2,
        "rated_rows": 1,
        "unique_examples": 1,
        "unique_evaluators": 1,
        "fabrication_flags": 0,
        "average_relevance_1_to_5": 4.0,
        "average_faithfulness_1_to_5": 5.0,
        "average_clarity_usefulness_1_to_5": 4.0,
        "average_overall_preference_1_to_5": 4.0,
        "common_qualitative_issues": ["Clear suggestions."],
    }

    csv_path, markdown_path = write_manual_evaluation_summary(summary, tmp_path)

    assert csv_path.exists()
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "# Phase 7 Manual Evaluation Summary" in markdown
    assert "Rated rows: 1" in markdown
    assert "Clear suggestions." in markdown
