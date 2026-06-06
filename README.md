# FitMyResume

FitMyResume is CS455 term project for building an end-to-end resume and job-description matching system. The planned system scores resume-job fit, explains matches and gaps, and rewrites resumes without fabricating experience.

## Project Structure

```text
data/
  raw/          Original downloaded datasets. Not committed to Git.
  interim/      Intermediate cleaned or merged files. Not committed to Git.
  processed/    Final train/validation/test files. Not committed to Git.
docs/           Project planning and documentation.
models/         LoRA adapter files; large adapter weights are tracked with Git LFS.
notebooks/      Exploration, experiments, and demo notebooks.
prompts/        Teacher, judge, and zero-shot prompts.
results/        Metrics, plots, tables, and error-analysis outputs.
src/            Reusable project scripts.
```

The final Qwen LoRA adapter is included under `models/qwen25-7b-fitmyresume-lora-v2/final/`;
large adapter weights are tracked with Git LFS.

## Final Results Artifacts

- Test-set score comparison: `results/all_methods_test_results.jsonl`
- Automatic evaluation summary: `results/evaluation/phase6_summary.md`
- Error analysis: `results/error_analysis/phase8_error_analysis_summary.md`
- Manual evaluation: `results/manual_evaluation/phase7_manual_evaluation_summary.md`
- Demo notebook: `notebooks/fitmyresume_demo.ipynb`
- Final all-methods test notebook: `notebooks/evaluate_all_methods_test.ipynb`

## Setup

Create and activate a Python environment, then install the project dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The default `requirements.txt` is the lightweight local/test environment. It is
enough for preprocessing, BM25, saved-artifact evaluation, manual-evaluation
summaries, and the unit tests.

GPU-backed model inference and sentence-transformer regeneration need the
optional model stack:

```powershell
python -m pip install -r requirements-gpu.txt
```

Install `requirements-gpu.txt` in Colab/Kaggle or another compatible GPU
runtime. `vllm` is mainly for Linux GPU serving; on Windows or unstable Colab
sessions, use the Transformers/PEFT fallback script instead.

## Run Tests

The local test suite is designed for the lightweight dependency set:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest
```

## Kaggle Credentials

To download Kaggle datasets, each team member needs their own Kaggle API token.

1. Open Kaggle account settings.
2. Go to the API section.
3. Generate a new API token.
4. Set it as an environment variable before running Kaggle CLI commands:

```powershell
$env:KAGGLE_API_TOKEN="paste-your-token-here"
```

This sets the token for the current PowerShell session. If you open a new terminal, set it again before running Kaggle commands.

Optionally, you can store the token in a local `.env` file for your own machine, but do not commit `.env` to Git.

## Download Dataset

From the project root, run:

```powershell
kaggle datasets download -d pranavvenugo/resume-and-job-description -p data/raw/kaggle --unzip
```

After download, the raw Kaggle files should be located at:

```text
data/raw/kaggle/Resume.csv
data/raw/kaggle/training_data.csv
```

Expected columns:

```text
Resume.csv:
ID, Resume_str, Resume_html, Category

training_data.csv:
company_name, job_description, position_title, description_length, model_response
```

## Preprocess Dataset

After the raw Kaggle files are available, run the deterministic preprocessing script:

```powershell
python src/preprocess_data.py --raw-dir data/raw/kaggle --interim-dir data/interim --processed-dir data/processed --seed 42
```

This creates cleaned interim files:

```text
data/interim/resumes_clean.csv
data/interim/jobs_clean.csv
```

And reproducible 80/10/10 split files:

```text
data/processed/resumes_train.csv
data/processed/resumes_validation.csv
data/processed/resumes_test.csv
data/processed/jobs_train.csv
data/processed/jobs_validation.csv
data/processed/jobs_test.csv
data/processed/preprocessing_metadata.json
```

The generated CSV files are not tracked by Git. Teammates should get the same outputs when they use the same raw Kaggle files and the same seed.

## Create Teacher Pairs

After preprocessing, create balanced resume-job pairs for teacher output generation:

```powershell
python src/create_teacher_pairs.py --split train --split validation --pairs-per-resume 3 --seed 42
```

For a small pilot:

```powershell
python src/create_teacher_pairs.py --split train --max-resumes 10 --seed 42
```

This writes:

```text
data/processed/teacher_pairs_train.csv
data/processed/teacher_pairs_validation.csv
```

Existing resume/job split CSVs are read only and are not modified.

## Run DeepSeek Teacher Pilot

Before generating teacher outputs for the full dataset, run `prompts/teacher_gold_output_prompt_v3.md` on a very small sample and inspect the result.

Create a DeepSeek API key, then add it to your local `.env` file:

```text
DEEPSEEK_API_KEY=your_deepseek_api_key_here
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

First run a dry run to confirm the prompt and data load correctly without calling DeepSeek:

```powershell
python src/run_teacher_pilot.py --split train --limit 1 --dry-run
```

Then run the real DeepSeek teacher pilot:

```powershell
python src/run_teacher_pilot.py --split train --limit 1
```

Pilot outputs are written to:

```text
results/teacher_pilot/
```

Inspect the JSON output before increasing `--limit` or generating the full teacher-labeled dataset.

## Build Instruction-Tuning Corpus

After teacher outputs are generated, build the fine-tuning JSONL by joining the
teacher outputs back to the source resume-job pairs:

```powershell
python src/build_instruction_corpus.py `
  --pairs-path data/processed/teacher_pairs_train.csv `
  --teacher-outputs-path results/teacher_pilot/deepseek_teacher_pilot_outputs.jsonl `
  --output-path data/instruction_tuning/instruction_tuning_train.jsonl `
  --skip-invalid
```

After validation teacher outputs are generated, build the validation corpus the
same way:

```powershell
python src/build_instruction_corpus.py `
  --pairs-path data/processed/teacher_pairs_validation.csv `
  --teacher-outputs-path results/teacher_validation/deepseek_teacher_pilot_outputs.jsonl `
  --output-path data/instruction_tuning/instruction_tuning_validation.jsonl `
  --skip-invalid
```

Use `--skip-invalid` only after reviewing the skipped count. Without that flag,
the script fails on the first malformed teacher output.

Each output row contains:

```text
instruction
input
output
metadata
```

The generated instruction-tuning files are derived artifacts and are not tracked
by Git.

## Fine-Tuned Qwen Inference

Phase 5 inference can use the trained Qwen2.5-7B LoRA adapter behind a vLLM
OpenAI-compatible server when the runtime supports it. See the Colab runbook for
the A100 commands:

```text
docs/phase5_finetuned_qwen_vllm_runbook.md
```

If Colab makes vLLM serving unreliable, use the Transformers/PEFT fallback script
to load the base model and LoRA adapter directly.

The reusable scripts are:

```text
src/run_finetuned_vllm_inference.py
src/run_finetuned_transformers_inference.py
src/summarize_finetuned_outputs.py
```

## Notebook Demo

Phase 9 uses a notebook demo instead of a Gradio app so the presentation path is
reliable even when local GPU serving is unavailable:

```text
notebooks/fitmyresume_demo.ipynb
```

The notebook defaults to the saved fine-tuned validation sample:

```text
results/finetuned_qwen_validation_transformers_sample50_outputs.jsonl
```

Open the notebook from the project checkout and run the cells top to bottom. It
loads a parseable fine-tuned output, shows the original resume/job-description
inputs, checks input lengths, and displays the fit score, explanation, and
resume suggestions. Optional cells document the vLLM and Transformers/PEFT live
inference commands for GPU-backed demos.

## Manual Evaluation

Phase 7 prepares a small human-review packet for generated resume suggestions.
Because full fine-tuned test-set outputs are not currently available in the
repository, the checked-in packet uses the available 50-example validation
sample and labels it accordingly.

Build the reviewer form and Markdown packet:

```powershell
python src/build_manual_evaluation.py `
  --instructions data/instruction_tuning/instruction_tuning_validation.jsonl `
  --outputs results/finetuned_qwen_validation_transformers_sample50_outputs.jsonl `
  --output-dir results/manual_evaluation `
  --sample-size 10 `
  --seed 42 `
  --sample-label validation_sample_finetuned_qwen
```

This writes:

```text
results/manual_evaluation/manual_evaluation_form.csv
results/manual_evaluation/manual_evaluation_packet.md
```

After at least one team member fills the rating columns in
`manual_evaluation_form.csv`, summarize the manual results:

```powershell
python src/summarize_manual_evaluation.py `
  --form results/manual_evaluation/manual_evaluation_form.csv `
  --output-dir results/manual_evaluation
```

This writes:

```text
results/manual_evaluation/manual_evaluation_summary_template.csv
results/manual_evaluation/phase7_manual_evaluation_summary.md
```

## Project Plan

See `docs/Chronological Project Tasks.md` for the chronological task list and project milestones.
