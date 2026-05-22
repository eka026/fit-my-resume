# Teacher Pairing Design

Date: 2026-05-22

## Purpose

`src/create_teacher_pairs.py` will create traceable resume-job pairs for Gemini teacher labeling. The raw Kaggle files do not share a join key, so teacher examples should not be created by pairing resume and job rows by index.

The paired dataset should contain a balanced range of likely fit quality so the teacher model can produce useful `score`, `explanation`, and `rewritten_resume` outputs for later instruction tuning.

## Inputs

The script reads existing processed split files:

- `data/processed/resumes_train.csv`
- `data/processed/jobs_train.csv`
- `data/processed/resumes_validation.csv`
- `data/processed/jobs_validation.csv`

It should support CLI options for:

- `--split`, repeatable or comma-separated, defaulting to `train,validation`
- `--pairs-per-resume`, defaulting to `3`
- `--max-resumes`, optional pilot cap
- `--seed`, defaulting to `42`
- `--processed-dir`, defaulting to `data/processed`

## Pairing Strategy

For each split, resumes are paired only with jobs from the same split. This prevents training/validation leakage.

For each resume:

1. Compute TF-IDF cosine similarity between the resume text and every job description in the same split.
2. Identify keyword-related jobs by comparing the resume category against job title and job description text.
3. Create up to three pair types:
   - `strong_hybrid`: a seeded random choice from high-similarity jobs, preferring category/title keyword-related jobs when available.
   - `medium_tfidf`: a seeded random choice from the middle similarity band.
   - `weak_random`: a seeded random choice from low-similarity or unrelated jobs.

Seeded randomness should choose among candidate bands instead of always taking the single top, middle, or bottom job. This keeps examples less mechanical and reduces overfitting risk.

The pairing strategy is not a ground-truth label. Gemini will still assign the final fit score.

## Outputs

The script writes one CSV per split:

- `data/processed/teacher_pairs_train.csv`
- `data/processed/teacher_pairs_validation.csv`

Required columns:

- `pair_id`
- `split`
- `resume_id`
- `job_id`
- `pairing_strategy`
- `similarity_score`
- `resume_category`
- `job_position_title`
- `resume_text`
- `job_description`

`pair_id` should be deterministic and include the split, resume ID, job ID, and strategy.

## Safety Rules

- Do not modify existing resume/job split CSVs.
- Do not pair across splits.
- Do not create duplicate `(resume_id, job_id)` rows within a split output.
- Keep output deterministic for the same seed and inputs.
- If keyword matching finds no candidates, fall back to TF-IDF candidate bands.
- Use `--max-resumes` for small pilot files before generating full outputs.

## Tests

Add focused tests that verify:

- required columns are present
- no duplicate `(resume_id, job_id)` pairs are produced
- output is deterministic with the same seed
- strong, medium, and weak strategies are present on a small synthetic dataset
