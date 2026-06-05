import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from src.run_finetuned_vllm_inference import (
    DEFAULT_BASE_MODEL,
    DEFAULT_MODEL,
    build_messages,
    build_output_row,
    read_jsonl,
    write_jsonl,
)


SERVING_BACKEND = "transformers_peft"


def render_chat_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        try:
            return str(
                apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        except ValueError as error:
            if "chat_template" not in str(error):
                raise

    rendered_messages = []
    for message in messages:
        role = message.get("role", "user").upper()
        content = message.get("content", "")
        rendered_messages.append(f"{role}:\n{content}")
    return "\n\n".join(rendered_messages) + "\n\nASSISTANT:\n"


def _optional_torch_no_grad():
    try:
        import torch
    except ImportError:
        return nullcontext()
    return torch.no_grad()


def call_transformers_chat(
    *,
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> str:
    prompt = render_chat_prompt(tokenizer, messages)
    inputs = tokenizer(prompt, return_tensors="pt")
    model_device = getattr(model, "device", None)
    if model_device is not None and hasattr(inputs, "to"):
        inputs = inputs.to(model_device)

    input_ids = inputs["input_ids"]
    prompt_token_count = len(input_ids[0])
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)

    generation_kwargs: dict[str, Any] = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": pad_token_id,
    }
    if temperature > 0:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

    with _optional_torch_no_grad():
        output_ids = model.generate(**generation_kwargs)

    generated_token_ids = output_ids[0][prompt_token_count:]
    return str(tokenizer.decode(generated_token_ids, skip_special_tokens=True)).strip()


def _resolve_torch_dtype(dtype: str) -> Any:
    if dtype == "auto":
        return "auto"
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required for transformers inference.") from error
    try:
        return getattr(torch, dtype)
    except AttributeError as error:
        raise ValueError(f"Unknown torch dtype: {dtype}") from error


def _default_tokenizer_source(*, base_model: str, adapter_path: str) -> str:
    if adapter_path:
        adapter_path_obj = Path(adapter_path)
        if adapter_path_obj.exists() and (adapter_path_obj / "tokenizer_config.json").exists():
            return adapter_path
    return base_model


def load_model_and_tokenizer(
    *,
    base_model: str,
    adapter_path: str,
    tokenizer_name_or_path: str,
    dtype: str,
    device_map: str,
    load_in_4bit: bool,
) -> tuple[Any, Any]:
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Install transformers and peft before running this script. "
            "For Colab, use: pip install -U transformers peft accelerate bitsandbytes"
        ) from error

    model_kwargs: dict[str, Any] = {
        "device_map": device_map,
        "torch_dtype": _resolve_torch_dtype(dtype),
    }
    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as error:
            raise RuntimeError("bitsandbytes quantization requires a transformers build with BitsAndBytesConfig.") from error
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=_resolve_torch_dtype(dtype if dtype != "auto" else "bfloat16"),
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    tokenizer_source = tokenizer_name_or_path or _default_tokenizer_source(
        base_model=base_model,
        adapter_path=adapter_path,
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True, **model_kwargs)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def run_inference(
    *,
    input_path: Path,
    output_path: Path,
    model: Any,
    tokenizer: Any,
    requested_model: str,
    base_model: str,
    adapter_name: str,
    adapter_path: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    limit: int | None,
    sleep_seconds: float,
) -> dict[str, int]:
    rows = read_jsonl(input_path)
    if limit is not None:
        rows = rows[:limit]

    generation_config = {
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
    }
    output_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        messages = build_messages(row)
        response_text = call_transformers_chat(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )
        output_row = build_output_row(
            corpus_row=row,
            response_text=response_text,
            model=requested_model,
            base_model=base_model,
            adapter_name=adapter_name,
            adapter_path=adapter_path,
            generation_config=generation_config,
        )
        output_row["serving_backend"] = SERVING_BACKEND
        output_rows.append(output_row)
        print(f"processed {index}/{len(rows)}: {output_rows[-1]['pair_id']}")
        if sleep_seconds > 0 and index < len(rows):
            time.sleep(sleep_seconds)

    write_jsonl(output_path, output_rows)
    parse_success = sum(1 for row in output_rows if row["parse_success"])
    return {
        "rows": len(output_rows),
        "parse_success": parse_success,
        "parse_failed": len(output_rows) - parse_success,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fine-tuned Qwen LoRA inference directly with transformers and PEFT."
    )
    parser.add_argument("--input", type=Path, required=True, help="Instruction-tuning JSONL input.")
    parser.add_argument("--output", type=Path, required=True, help="JSONL path for model outputs.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model label to record in outputs.")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter-name", default=DEFAULT_MODEL)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument(
        "--tokenizer",
        default="",
        help="Optional tokenizer path/name. Defaults to adapter tokenizer when present, otherwise base model.",
    )
    parser.add_argument("--dtype", default="bfloat16", help="Torch dtype name, such as bfloat16, float16, or auto.")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--load-in-4bit", action="store_true", help="Use bitsandbytes 4-bit loading.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for smoke tests.")
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="Optional delay between generations.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    model, tokenizer = load_model_and_tokenizer(
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        tokenizer_name_or_path=args.tokenizer,
        dtype=args.dtype,
        device_map=args.device_map,
        load_in_4bit=args.load_in_4bit,
    )
    summary = run_inference(
        input_path=args.input,
        output_path=args.output,
        model=model,
        tokenizer=tokenizer,
        requested_model=args.model,
        base_model=args.base_model,
        adapter_name=args.adapter_name,
        adapter_path=args.adapter_path,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        limit=args.limit,
        sleep_seconds=args.sleep_seconds,
    )
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
