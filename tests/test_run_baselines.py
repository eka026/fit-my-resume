import json

from src.run_baselines import bm25_scores, generate_bm25_outputs


def test_bm25_scores_rank_matching_document_highest():
    scores = bm25_scores(
        query="python sql machine learning",
        documents=[
            "python machine learning sql models",
            "forklift warehouse inventory",
        ],
    )

    assert scores[0] > scores[1]


def test_generate_bm25_outputs_writes_prediction_rows(tmp_path):
    input_path = tmp_path / "validation.jsonl"
    output_path = tmp_path / "bm25.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "input": "RESUME:\npython sql machine learning\n\nJOB_DESCRIPTION:\npython sql models",
                "metadata": {
                    "pair_id": "pair_1",
                    "split": "validation",
                    "resume_id": "resume_1",
                    "job_id": "job_1",
                    "pairing_strategy": "strong_hybrid",
                },
                "output": json.dumps({"score": 80}),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows = generate_bm25_outputs(input_path, output_path)

    written = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert rows == written
    assert written[0]["pair_id"] == "pair_1"
    assert written[0]["model"] == "bm25"
    assert 0 <= written[0]["pred_score"] <= 100
