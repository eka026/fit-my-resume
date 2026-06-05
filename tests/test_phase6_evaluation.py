import json

import pytest

from src.evaluate_outputs import (
    evaluate_score_predictions,
    evaluate_systems,
    extract_explanation_text,
    load_teacher_references,
    rouge_l_f1,
)


def test_rouge_l_f1_rewards_longest_common_subsequence_overlap():
    score = rouge_l_f1(
        "managed python sql models",
        "managed python models",
    )

    assert score == pytest.approx(85.7143)


def test_extract_explanation_text_handles_structured_and_string_explanations():
    structured = {
        "matched_qualifications": ["Python", "SQL"],
        "missing_or_weak_qualifications": ["Cloud"],
        "overall_reasoning": "Good technical overlap.",
    }

    assert extract_explanation_text(structured) == "Python SQL Cloud Good technical overlap."
    assert extract_explanation_text("Plain explanation.") == "Plain explanation."


def test_load_teacher_references_reads_teacher_output_jsonl(tmp_path):
    path = tmp_path / "teacher.jsonl"
    path.write_text(
        json.dumps(
            {
                "pair_id": "pair_1",
                "teacher_output": {
                    "score": 70,
                    "explanation": "Strong fit.",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    references = load_teacher_references(path)

    assert references["pair_1"].score == 70.0
    assert references["pair_1"].explanation_text == "Strong fit."


def test_evaluate_score_predictions_reports_regression_and_binary_metrics():
    rows = [
        {"pair_id": "pair_1", "pred_score": 80},
        {"pair_id": "pair_2", "pred_score": 20},
        {"pair_id": "pair_3", "pred_score": 75},
    ]
    references = {
        "pair_1": 90,
        "pair_2": 10,
        "pair_3": 40,
    }

    metrics = evaluate_score_predictions(rows, references, threshold=50)

    assert metrics["evaluated_count"] == 3
    assert metrics["mae"] == pytest.approx(18.3333)
    assert metrics["rmse"] == pytest.approx(21.7945)
    assert metrics["accuracy"] == pytest.approx(66.6667)
    assert metrics["macro_f1"] == pytest.approx(66.6667)


def test_evaluate_systems_keeps_score_and_explanation_counts_separate(tmp_path):
    teacher_path = tmp_path / "teacher.jsonl"
    system_path = tmp_path / "bm25.jsonl"
    output_dir = tmp_path / "evaluation"
    teacher_path.write_text(
        json.dumps(
            {
                "pair_id": "pair_1",
                "teacher_output": {
                    "score": 70,
                    "explanation": "Python SQL role match.",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    system_path.write_text(
        json.dumps({"pair_id": "pair_1", "pred_score": 65}) + "\n",
        encoding="utf-8",
    )

    rows = evaluate_systems(
        teacher_path=teacher_path,
        system_paths={"bm25": system_path},
        output_dir=output_dir,
        threshold=50,
    )

    assert rows[0]["score_evaluated_count"] == 1
    assert rows[0]["explanation_evaluated_count"] == 0
