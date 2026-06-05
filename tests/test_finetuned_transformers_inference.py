import json
from pathlib import Path

from src.run_finetuned_transformers_inference import (
    call_transformers_chat,
    render_chat_prompt,
    run_inference,
)


class FakeInputs(dict):
    def to(self, device):
        self["device"] = device
        return self


class FakeTokenizer:
    eos_token_id = 99
    pad_token_id = None

    def __init__(self):
        self.chat_template_args = None
        self.encoded_prompt = None

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        self.chat_template_args = {
            "messages": messages,
            "tokenize": tokenize,
            "add_generation_prompt": add_generation_prompt,
        }
        return "SYSTEM: Return JSON only.\nUSER: Resume text.\nASSISTANT:"

    def __call__(self, prompt, return_tensors):
        self.encoded_prompt = {"prompt": prompt, "return_tensors": return_tensors}
        return FakeInputs({"input_ids": [[1, 2, 3]]})

    def decode(self, tokens, skip_special_tokens):
        assert tokens == [4, 5]
        assert skip_special_tokens is True
        return '{"score": 82, "explanation": {}, "resume_suggestions": []}'


class FakeModel:
    device = "cpu"

    def __init__(self):
        self.generate_kwargs = None

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return [[1, 2, 3, 4, 5]]


def test_render_chat_prompt_uses_tokenizer_chat_template():
    tokenizer = FakeTokenizer()
    messages = [
        {"role": "system", "content": "Return JSON only."},
        {"role": "user", "content": "Resume text."},
    ]

    prompt = render_chat_prompt(tokenizer, messages)

    assert prompt.endswith("ASSISTANT:")
    assert tokenizer.chat_template_args == {
        "messages": messages,
        "tokenize": False,
        "add_generation_prompt": True,
    }


def test_call_transformers_chat_decodes_only_new_tokens():
    tokenizer = FakeTokenizer()
    model = FakeModel()

    response = call_transformers_chat(
        model=model,
        tokenizer=tokenizer,
        messages=[{"role": "user", "content": "Resume text."}],
        temperature=0.0,
        top_p=1.0,
        max_new_tokens=128,
    )

    assert json.loads(response)["score"] == 82
    assert model.generate_kwargs["do_sample"] is False
    assert model.generate_kwargs["max_new_tokens"] == 128
    assert model.generate_kwargs["pad_token_id"] == tokenizer.eos_token_id


def test_run_inference_writes_vllm_compatible_output_rows(tmp_path):
    input_path = Path(tmp_path) / "input.jsonl"
    output_path = Path(tmp_path) / "outputs.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "instruction": "Return JSON only.",
                "input": "RESUME:\nBuilt APIs.\n\nJOB_DESCRIPTION:\nNeeds API work.",
                "output": "{\"score\": 75}",
                "metadata": {
                    "pair_id": "validation_resume_1_job_1",
                    "split": "validation",
                    "prompt_version": "teacher_gold_output_prompt_v3",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = run_inference(
        input_path=input_path,
        output_path=output_path,
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        requested_model="fitmyresume",
        base_model="Qwen/Qwen2.5-7B-Instruct",
        adapter_name="fitmyresume",
        adapter_path="/content/adapter/final",
        temperature=0.0,
        top_p=1.0,
        max_new_tokens=128,
        limit=None,
        sleep_seconds=0.0,
    )

    output_row = json.loads(output_path.read_text(encoding="utf-8").strip())
    assert summary == {"rows": 1, "parse_success": 1, "parse_failed": 0}
    assert output_row["pair_id"] == "validation_resume_1_job_1"
    assert output_row["model"] == "fitmyresume"
    assert output_row["serving_backend"] == "transformers_peft"
    assert output_row["generation_config"]["max_new_tokens"] == 128
    assert output_row["parsed_output"]["score"] == 82
