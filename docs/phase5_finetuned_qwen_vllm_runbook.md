# Phase 5 Fine-Tuned Qwen vLLM Runbook

This runbook completes Phase 5 after the Qwen2.5-7B LoRA adapter has been trained.
Run these commands in Colab with an A100 runtime unless noted otherwise.

## Expected Artifacts

The trained adapter should exist at:

```text
/content/drive/MyDrive/fit-my-resume/models/qwen25-7b-fitmyresume-lora-v2/final
```

The repo should contain or be able to create:

```text
data/instruction_tuning/instruction_tuning_validation.jsonl
data/instruction_tuning/instruction_tuning_test.jsonl
```

The validation corpus already exists in the local repo. If the test corpus is missing in Colab, build it before running test inference.

## 1. Install Serving Dependencies

```bash
pip install -U vllm
```

If Colab restarts after installing CUDA-related packages, rerun the notebook cells that mount Drive and enter the repo directory.

## 2. Optional: Build the Test Instruction Corpus

Run this only if `data/instruction_tuning/instruction_tuning_test.jsonl` does not exist.

```bash
python src/build_instruction_corpus.py \
  --pairs-path data/processed/teacher_pairs_test.csv \
  --teacher-outputs-path results/teacher_test/deepseek_teacher_pilot_outputs.jsonl \
  --output-path data/instruction_tuning/instruction_tuning_test.jsonl \
  --skip-invalid
```

Review the printed skipped/invalid counts. If the invalid count is unexpectedly high, inspect the teacher test outputs before proceeding.

## 3. Start vLLM With the Fine-Tuned LoRA Adapter

Run this in a dedicated Colab cell and leave it running.

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --enable-lora \
  --lora-modules fitmyresume=/content/drive/MyDrive/fit-my-resume/models/qwen25-7b-fitmyresume-lora-v2/final \
  --max-lora-rank 32
```

The LoRA adapter name is `fitmyresume`. The inference script uses that name as the requested model.

## 4. Smoke Test Fine-Tuned Inference

Run this in a second Colab cell or terminal while the vLLM server is still running.

```bash
python src/run_finetuned_vllm_inference.py \
  --input data/instruction_tuning/instruction_tuning_validation.jsonl \
  --output results/finetuned_qwen_validation_smoke_outputs.jsonl \
  --model fitmyresume \
  --base-url http://localhost:8000/v1 \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --adapter-name fitmyresume \
  --adapter-path /content/drive/MyDrive/fit-my-resume/models/qwen25-7b-fitmyresume-lora-v2/final \
  --limit 5
```

Summarize the smoke-test outputs:

```bash
python src/summarize_finetuned_outputs.py \
  --input results/finetuned_qwen_validation_smoke_outputs.jsonl \
  --summary-output results/finetuned_qwen_validation_smoke_summary.json
```

Manually inspect the first few output rows and confirm:

- `parse_success` is usually `true`.
- `parsed_output` contains `score`, `explanation`, and `resume_suggestions`.
- Suggestions do not fabricate experience.

## 5. Run Full Validation Inference

```bash
python src/run_finetuned_vllm_inference.py \
  --input data/instruction_tuning/instruction_tuning_validation.jsonl \
  --output results/finetuned_qwen_validation_outputs.jsonl \
  --model fitmyresume \
  --base-url http://localhost:8000/v1 \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --adapter-name fitmyresume \
  --adapter-path /content/drive/MyDrive/fit-my-resume/models/qwen25-7b-fitmyresume-lora-v2/final
```

```bash
python src/summarize_finetuned_outputs.py \
  --input results/finetuned_qwen_validation_outputs.jsonl \
  --summary-output results/finetuned_qwen_validation_summary.json
```

## 6. Run Full Test Inference

```bash
python src/run_finetuned_vllm_inference.py \
  --input data/instruction_tuning/instruction_tuning_test.jsonl \
  --output results/finetuned_qwen_test_outputs.jsonl \
  --model fitmyresume \
  --base-url http://localhost:8000/v1 \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --adapter-name fitmyresume \
  --adapter-path /content/drive/MyDrive/fit-my-resume/models/qwen25-7b-fitmyresume-lora-v2/final
```

```bash
python src/summarize_finetuned_outputs.py \
  --input results/finetuned_qwen_test_outputs.jsonl \
  --summary-output results/finetuned_qwen_test_summary.json
```

## 7. Phase 5 Completion Check

After these commands finish, Phase 5 can be marked complete if these files exist:

```text
results/finetuned_qwen_validation_outputs.jsonl
results/finetuned_qwen_validation_summary.json
results/finetuned_qwen_test_outputs.jsonl
results/finetuned_qwen_test_summary.json
```

Record the vLLM serving settings in the final report:

```text
base_model: Qwen/Qwen2.5-7B-Instruct
adapter_name: fitmyresume
adapter_path: /content/drive/MyDrive/fit-my-resume/models/qwen25-7b-fitmyresume-lora-v2/final
serving_backend: vLLM OpenAI-compatible server
lora_rank: 32
dtype: bfloat16
gpu: A100
```
