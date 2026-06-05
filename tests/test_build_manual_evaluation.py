import csv
import json

from src.build_manual_evaluation import (
    build_manual_evaluation_rows,
    write_manual_evaluation_artifacts,
)


def write_jsonl(path, rows):
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def instruction_row(pair_id, strategy, resume="Resume text", job="Job text"):
    return {
        "instruction": "Evaluate fit.",
        "input": f"RESUME:\n{resume}\n\nJOB_DESCRIPTION:\n{job}",
        "output": "{}",
        "metadata": {
            "pair_id": pair_id,
            "split": "validation",
            "resume_id": f"resume_{pair_id}",
            "job_id": f"job_{pair_id}",
            "pairing_strategy": strategy,
        },
    }


def output_row(pair_id, suggestions):
    return {
        "pair_id": pair_id,
        "parse_success": True,
        "parsed_output": {
            "score": 70,
            "resume_suggestions": suggestions,
        },
    }


def test_build_manual_evaluation_rows_joins_inputs_outputs_and_rating_columns(tmp_path):
    instructions_path = tmp_path / "instructions.jsonl"
    outputs_path = tmp_path / "outputs.jsonl"
    write_jsonl(
        instructions_path,
        [
            instruction_row(
                "pair_1",
                "strong_hybrid",
                resume="Python developer resume",
                job="Backend developer job",
            )
        ],
    )
    write_jsonl(
        outputs_path,
        [
            output_row(
                "pair_1",
                [
                    {
                        "section": "summary",
                        "action": "reword",
                        "suggestion": "Emphasize Python backend work.",
                        "evidence_from_resume": "Python developer resume",
                        "job_requirement_addressed": "Backend developer job",
                    }
                ],
            )
        ],
    )

    rows = build_manual_evaluation_rows(
        instructions_path=instructions_path,
        outputs_path=outputs_path,
        sample_size=1,
        seed=7,
        sample_label="validation_sample",
    )

    assert len(rows) == 1
    assert rows[0]["pair_id"] == "pair_1"
    assert rows[0]["sample_label"] == "validation_sample"
    assert rows[0]["resume_text"] == "Python developer resume"
    assert rows[0]["job_description"] == "Backend developer job"
    assert "Emphasize Python backend work." in rows[0]["generated_rewrite"]
    assert rows[0]["relevance_1_to_5"] == ""
    assert rows[0]["fabricated_content_flag"] == ""
    assert rows[0]["qualitative_notes"] == ""


def test_build_manual_evaluation_rows_balances_pairing_strategy_when_possible(tmp_path):
    instructions_path = tmp_path / "instructions.jsonl"
    outputs_path = tmp_path / "outputs.jsonl"
    instruction_rows = [
        instruction_row("pair_1", "strong_hybrid"),
        instruction_row("pair_2", "strong_hybrid"),
        instruction_row("pair_3", "medium_tfidf"),
        instruction_row("pair_4", "weak_random"),
    ]
    write_jsonl(instructions_path, instruction_rows)
    write_jsonl(
        outputs_path,
        [output_row(row["metadata"]["pair_id"], [{"suggestion": row["metadata"]["pairing_strategy"]}]) for row in instruction_rows],
    )

    rows = build_manual_evaluation_rows(
        instructions_path=instructions_path,
        outputs_path=outputs_path,
        sample_size=3,
        seed=3,
        sample_label="validation_sample",
    )

    assert {row["pairing_strategy"] for row in rows} == {
        "strong_hybrid",
        "medium_tfidf",
        "weak_random",
    }


def test_write_manual_evaluation_artifacts_outputs_csv_and_markdown(tmp_path):
    rows = [
        {
            "sample_label": "validation_sample",
            "pair_id": "pair_1",
            "split": "validation",
            "resume_id": "resume_1",
            "job_id": "job_1",
            "pairing_strategy": "strong_hybrid",
            "resume_text": "Resume text",
            "job_description": "Job text",
            "generated_rewrite": "Generated suggestions",
            "evaluator_id": "",
            "relevance_1_to_5": "",
            "faithfulness_1_to_5": "",
            "clarity_usefulness_1_to_5": "",
            "overall_preference_1_to_5": "",
            "fabricated_content_flag": "",
            "qualitative_notes": "",
        }
    ]

    csv_path, markdown_path = write_manual_evaluation_artifacts(rows, tmp_path)

    with csv_path.open(encoding="utf-8", newline="") as file:
        written = list(csv.DictReader(file))
    assert written[0]["pair_id"] == "pair_1"
    assert written[0]["relevance_1_to_5"] == ""

    markdown = markdown_path.read_text(encoding="utf-8")
    assert "# Phase 7 Manual Evaluation Packet" in markdown
    assert "pair_1" in markdown
    assert "Generated suggestions" in markdown
