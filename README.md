# FitMyResume

FitMyResume is CS455 term project for building an end-to-end resume and job-description matching system. The planned system scores resume-job fit, explains matches and gaps, and rewrites resumes without fabricating experience.

## Project Structure

```text
data/
  raw/          Original downloaded datasets. Not committed to Git.
  interim/      Intermediate cleaned or merged files. Not committed to Git.
  processed/    Final train/validation/test files. Not committed to Git.
docs/           Project planning and documentation.
models/         Local model checkpoints or LoRA adapters. Not committed to Git.
notebooks/      Exploration, experiments, and demo notebooks.
prompts/        Teacher, judge, and zero-shot prompts.
results/        Metrics, plots, tables, and error-analysis outputs.
src/            Reusable project scripts.
```

## Setup

Create and activate a Python environment, then install the project dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The current `requirements.txt` includes the Kaggle CLI used for downloading the dataset.

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
## Project Plan

See `docs/Chronological Project Tasks.md` for the chronological task list and project milestones.
