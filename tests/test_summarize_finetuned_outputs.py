import json

from src.summarize_finetuned_outputs import summarize_outputs


def test_summarize_outputs_counts_parse_rate_and_score_coverage(tmp_path):
    output_path = tmp_path / "finetuned_outputs.jsonl"
    rows = [
        {
            "pair_id": "pair_1",
            "parse_success": True,
            "parsed_output": {"score": 80},
        },
        {
            "pair_id": "pair_2",
            "parse_success": True,
            "parsed_output": {"score": 40},
        },
        {
            "pair_id": "pair_3",
            "parse_success": False,
            "parsed_output": None,
            "parse_error": "No valid JSON object",
        },
    ]
    output_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    summary = summarize_outputs(output_path)

    assert summary == {
        "rows": 3,
        "parse_success": 2,
        "parse_failed": 1,
        "parse_rate": 66.6667,
        "score_count": 2,
        "score_min": 40.0,
        "score_max": 80.0,
        "score_mean": 60.0,
    }
