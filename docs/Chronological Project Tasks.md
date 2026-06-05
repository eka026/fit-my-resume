# FitMyResume Chronological Task Plan

**Goal:** Complete the FitMyResume CS455 project: an end-to-end resume-job matching system that scores fit, explains matches/gaps, and rewrites resumes without fabricating experience.

**Core deliverables by the end of the project:**
- Fine-tuned Qwen2.5-7B-Instruct model using QLoRA/LoRA adapters, or a documented smaller-model fallback if compute prevents 7B training.
- Fine-tuned inference pipeline that accepts a resume and job description and returns valid JSON with `score`, `explanation`, and `resume_suggestions`, using vLLM when the runtime supports it or a Transformers/PEFT fallback when Colab serving is unstable.
- Baseline comparison against BM25, sentence-transformer similarity, and zero-shot Qwen2.5-7B-Instruct.
- Automatic evaluation results plus a small manual evaluation component for rewritten resumes.
- Error analysis, final report, cleaned repository, and short demo notebook or Gradio interface.

**Important suggestions incorporated:**
- Add a small manual evaluation of rewritten resumes, not only LLM-as-judge.
- Spot-check teacher-generated outputs thoroughly before using them for fine-tuning.
- Prepare compute fallbacks and optimize training so Colab/Kaggle limits do not block the project.
- Use vLLM for high-throughput model serving when the runtime supports it; use direct Transformers/PEFT inference as the documented fallback path. Do not use vLLM for QLoRA training itself.

---

## Phase 1 - Environment Setup and Data Engineering

Original proposal window: Week 1, now to May 10.

- [X] Set up the shared GitHub repository.
- [X] Add a clear repository structure:
  - `data/` for raw, interim, and processed dataset files.
  - `notebooks/` for exploration and demo notebooks.
  - `src/` for reusable scripts.
  - `prompts/` for teacher, judge, and zero-shot prompts.
  - `results/` for metrics, tables, plots, and error-analysis samples.
  - `models/` or external storage notes for LoRA adapters and checkpoints.
- [X] Set up Colab Pro, Colab free, and Kaggle Notebook environments.
- [X] Verify access to required services:
  - Kaggle dataset download.
  - Hugging Face model access for Qwen2.5-7B-Instruct.
  - Teacher LLM API access.
  - Judge LLM API access.
- [X] Download the "Resume and Job Description" Kaggle dataset.
- [X] Inspect dataset columns and confirm the available fields:
  - Resume text.
  - Job description text.
  - Match labels or category labels.
  - Any IDs or metadata needed for joins.
- [X] Build the initial preprocessing script.
- [X] Remove or mask obvious personally identifiable information:
  - Names.
  - Emails.
  - Phone numbers.
  - Addresses.
  - URLs if they identify a person.
- [X] Clean noisy text:
  - Remove broken characters.
  - Normalize whitespace.
  - Normalize date formatting where practical.
  - Drop empty, duplicate, or unusable rows.
- [X] Create train, validation, and test splits using an 80/10/10 split.
- [X] Save split files with stable IDs so later generated outputs can be traced back.
- [X] Draft the teacher prompt for generating gold outputs.
- [X] Run the teacher prompt on a very small sample before generating the full dataset.

**Definition of done:** The cleaned dataset is split and saved, the environments run, and the teacher prompt is ready for quality testing.

---

## Phase 2 - Teacher Output Generation and Quality Control

Original proposal window: Week 2, May 11 to May 17.

- [X] Generate teacher outputs for a small pilot batch first.
  - [X] Created and ran a dry-run pilot artifact: `results/teacher_pilot/deepseek_teacher_pilot_dry_run.jsonl`.
  - [X] Run the real DeepSeek pilot after `DEEPSEEK_API_KEY` is available.
- [X] Validate that each teacher output contains valid JSON.
  - [X] Added pilot-output JSON parsing in `src/run_teacher_pilot.py`.
  - [X] Verified the pilot validation tests pass with pytest plugin autoload disabled.
- [X] Check that each output includes all required fields:
  - `score`.
  - `explanation`.
  - `resume_suggestions`.
  - [X] Added schema validation for required top-level fields in `src/run_teacher_pilot.py`.
  - [X] Added tests covering valid schema acceptance and invalid score rejection.
- [X] Manually spot-check the pilot outputs for quality.
- [X] Reject or revise outputs that:
  - Invent experience not present in the original resume.
  - Give scores inconsistent with the resume-job pair.
  - Produce vague explanations with no job-specific evidence.
  - Fail JSON parsing.
  - Rewrite the resume too aggressively.
- [X] Revise the teacher prompt if the pilot batch has recurring problems.
- [X] Add a small human validation step for teacher outputs:
  - Select a random sample of teacher-generated examples.
  - Have team members check factuality and usefulness.
  - Record the number of hallucinated, vague, or malformed outputs.
- [X] Generate teacher outputs for the full training and validation sets after the pilot passes.
- [X] Save teacher outputs with source IDs and prompt version.
- [X] Convert the dataset into instruction-tuning format:
  - Instruction: task definition and output constraints.
  - Input: resume plus job description.
  - Output: teacher-generated JSON.
- [X] Build a JSON validation script for the instruction-tuning corpus.
- [X] Run validation on the full generated corpus.
- [X] Remove or repair malformed examples before training.

**Definition of done:** The instruction-tuning dataset is valid, traceable, quality-checked, and ready for baseline and training work.

---

## Phase 3 - Baseline Systems

Complete before judging whether fine-tuning helped.

- [X] Implement BM25 keyword-matching baseline for score prediction.
- [X] Run BM25 on the validation and test sets.
- [X] Save BM25 scores and metrics.
- [X] Implement sentence-transformer cosine-similarity baseline using `all-mpnet-base-v2`.
- [X] Run sentence-transformer baseline on the validation and test sets.
- [X] Save sentence-transformer scores and metrics.
- [X] Implement zero-shot Qwen2.5-7B-Instruct prompting using the same JSON schema.
- [X] Serve the zero-shot Qwen model with vLLM if the available GPU/runtime supports it.
- [X] Run zero-shot Qwen on a small sample first to verify formatting.
- [x] Run zero-shot Qwen on the test set if compute permits.
- [X] Track JSON parse rate for zero-shot outputs.
- [X] Save all baseline outputs in a consistent results format.

**Definition of done:** The project has baseline results for comparison before fine-tuned model results are interpreted.

---

## Phase 4 - QLoRA Training Preparation

Original proposal window: Week 3, May 18 to May 24.

- [X] Create a small training smoke test using a tiny subset of the instruction-tuning data.
- [X] Load Qwen2.5-7B-Instruct in 4-bit quantization with `bitsandbytes`.
- [X] Attach LoRA adapters with the initial planned configuration:
  - Rank: 16.
  - Target modules: attention projection layers such as `q_proj` and `v_proj`.
- [X] Configure Hugging Face TRL `SFTTrainer`.
- [X] Use a formatting function or collator that trains on the expected answer tokens, not unnecessary prompt reconstruction.
- [X] Set generation and training constraints:
  - Maximum sequence length.
  - Batch size and gradient accumulation.
  - Learning rate.
  - Epoch count.
  - Checkpoint frequency.
- [X] Run the smoke test and confirm:
  - Training starts without memory failure.
  - Loss decreases on the small sample.
  - Checkpoints are saved.
  - Inference produces parseable JSON on a few examples.
- [X] Prepare compute fallback settings:
  - Smaller batch size.
  - More gradient accumulation.
  - Shorter context length.
  - Fewer training examples for a first complete run.
  - More frequent adapter checkpointing.
  - Smaller Qwen-family fallback if Qwen2.5-7B cannot run reliably.

**Definition of done:** The training script works on a small subset and the team knows the fallback path if full training hits compute limits.

---

## Phase 5 - Full Fine-Tuning and Inference

Run this after the training smoke test passes.

- [ ] Start full QLoRA fine-tuning on the prepared training set.
- [ ] Save checkpoints frequently to persistent storage.
- [ ] Monitor:
  - Training loss.
  - Validation loss.
  - GPU memory usage.
  - Runtime and interruption risk.
- [ ] Stop early if validation loss worsens or outputs degrade.
- [ ] Save the final LoRA adapter and configuration.
- [ ] Decide the primary inference backend for the trained model:
  - Merge the LoRA adapter into the base model and serve the merged model with vLLM, or
  - Serve the base model with LoRA adapter support if the selected vLLM version and runtime support it, or
  - Load the base model plus LoRA adapter directly with Transformers/PEFT if Colab makes vLLM unreliable.
- [X] Write or finalize the vLLM-based inference script.
- [X] Write the Transformers/PEFT fallback inference script.
- [ ] Run a Transformers/PEFT smoke test against the saved LoRA adapter in Colab.
- [ ] Start a local vLLM server for the selected model or fallback model when GPU/runtime support is available.
- [ ] Call the selected inference backend from the inference script.
- [ ] Run the fine-tuned model on validation examples.
- [ ] Check JSON parse rate.
- [ ] Check sample explanations and rewritten resumes for hallucinations.
- [ ] Run the fine-tuned model on the full test set.
- [ ] Save all generated test outputs with model version, adapter version, prompt version, serving backend, and backend configuration.

**Definition of done:** The fine-tuned model has produced test-set outputs through vLLM or the documented Transformers/PEFT fallback path, and those outputs can be evaluated against all baselines.

---

## Phase 6 - Automatic Evaluation

Original proposal window: Week 4, May 25 to May 31.

- [ ] Evaluate fit-score quality:
  - Pearson correlation against match labels.
  - Spearman correlation against match labels.
  - Accuracy if labels are binarized.
  - Macro F1 if labels are binarized.
- [ ] Evaluate explanation similarity:
  - ROUGE-L against teacher references.
  - BERTScore against teacher references.
- [ ] Build the LLM-as-judge rubric for explanations.
- [ ] Run judge evaluation with a model different from the teacher model.
- [ ] Score explanations on:
  - Correctness.
  - Specificity.
  - Coverage of matched qualifications.
  - Coverage of missing qualifications.
- [ ] Build pairwise judge prompts for rewritten resumes.
- [ ] Run pairwise comparison between:
  - Original resume and rewritten resume.
  - Zero-shot rewrite and fine-tuned rewrite.
- [ ] Run bidirectional pairwise comparisons to reduce position bias.
- [ ] Ask the judge to flag fabricated content in each rewrite.
- [ ] Save raw judge outputs and parsed metrics.
- [ ] Create summary tables comparing:
  - BM25.
  - Sentence-transformer.
  - Zero-shot Qwen.
  - Fine-tuned Qwen.

**Definition of done:** All automatic metrics are computed and summarized in tables ready for the final report.

---

## Phase 7 - Manual Evaluation Component

This phase is added because of the instructor's suggestion.

- [ ] Select a small manual evaluation sample from the test set.
- [ ] Include examples from different job categories when possible.
- [ ] Prepare a lightweight human evaluation form.
- [ ] For each sampled example, show evaluators:
  - Original resume.
  - Target job description.
  - Generated rewrite.
- [ ] Ask evaluators to score each rewrite from 1 to 5 on:
  - Relevance to the job.
  - Faithfulness to the original resume.
  - Clarity and usefulness.
  - Overall preference.
- [ ] Ask evaluators to mark whether the rewrite includes fabricated content.
- [ ] Have at least two team members evaluate the same sample if time allows.
- [ ] Summarize manual results:
  - Average score per criterion.
  - Number of fabrication flags.
  - Common qualitative issues.
- [ ] Compare manual results with LLM-as-judge results.
- [ ] Note agreements and disagreements between human and automatic evaluation.

**Definition of done:** The final report includes a small human evaluation section that complements the automatic judge.

---

## Phase 8 - Error Analysis and Iteration

Run after automatic and manual evaluation produce initial results.

- [ ] Sample weak examples from each system.
- [ ] Categorize failure modes:
  - Invalid JSON.
  - Wrong or poorly calibrated score.
  - Missing important job requirements.
  - Overly generic explanation.
  - Fabricated resume content.
  - Rewrite that removes important candidate evidence.
  - Prompt injection vulnerability.
- [ ] Identify whether failures come mostly from:
  - Dataset noise.
  - Teacher-output quality.
  - Prompt design.
  - Training settings.
  - Compute-constrained truncation.
  - Model limitations.
- [ ] If needed, revise prompts and rerun a targeted subset.
- [ ] If needed, clean or remove low-quality training examples.
- [ ] If needed, run a second fine-tuning attempt with adjusted settings.
- [ ] Document all changes so the final report can explain what was tried.

**Definition of done:** The team can explain where the system succeeds, where it fails, and why.

---

## Phase 9 - Ablations and Compute Fallback Experiments

Original proposal window: Week 5, June 1 to June 7.

- [ ] Decide which ablations are realistic within the remaining time.
- [ ] Prioritize the most informative experiments:
  - LoRA rank comparison, such as 8 vs. 16 vs. 32.
  - Smaller training subset vs. full training set.
  - Different learning rate.
  - Smaller Qwen-family fallback if Qwen2.5-7B is unstable.
- [ ] Run only ablations that can finish and be evaluated reliably.
- [ ] Keep the same validation/test split across experiments.
- [ ] Compare ablation results against the main fine-tuned model.
- [ ] Record compute cost and runtime for each experiment.
- [ ] Decide the final model checkpoint to present.

**Definition of done:** The report can honestly state whether fine-tuning improved results and under which settings it was worth the cost.

---

## Phase 10 - Demo and User Interface

Build after the final model or fallback model is selected.

- [ ] Decide demo format:
  - Gradio interface, or
  - Notebook demo if deployment time is limited.
- [ ] Run the selected final model behind a local vLLM server for the demo when GPU resources allow.
- [ ] If vLLM is not feasible in the demo environment, document the fallback inference path clearly.
- [ ] Implement input fields:
  - Resume text.
  - Job description text.
- [ ] Add output display sections:
  - Fit score.
  - Explanation.
  - Rewritten resume.
- [ ] Add JSON parsing and error handling.
- [ ] Add input length checks so long resumes do not crash inference.
- [ ] Test the demo on multiple examples.
- [ ] Prepare one polished demo example for the presentation.
- [ ] Save screenshots or short notes for the final report/presentation.

**Definition of done:** A stakeholder can run the demo and see the complete resume-job matching workflow, preferably through vLLM-backed inference or with a documented fallback if vLLM is unavailable.

---

## Phase 11 - Final Report, Repository Cleanup, and Submission

Complete this after experiments are frozen.

- [ ] Freeze final results.
- [ ] Create final result tables:
  - Score metrics.
  - Explanation metrics.
  - Rewrite judge results.
  - Manual evaluation results.
  - Ablation results.
- [ ] Create final figures if useful:
  - Metric comparison chart.
  - Ablation comparison chart.
  - Error category distribution.
- [ ] Write the final report sections:
  - Problem and motivation.
  - Dataset and preprocessing.
  - Teacher-data generation.
  - Baselines.
  - Fine-tuning method.
  - Evaluation setup.
  - Results.
  - Manual evaluation.
  - Error analysis.
  - Limitations.
  - Compute constraints and fallbacks.
  - Conclusion.
- [ ] Document repository usage:
  - Installation/setup instructions.
  - Data preparation instructions.
  - Training command.
  - vLLM serving command.
  - Transformers/PEFT fallback inference command.
  - Evaluation command.
  - Demo command.
- [ ] Remove secrets, API keys, and private files from the repository.
- [ ] Confirm notebooks run from top to bottom or clearly document required runtime assumptions.
- [ ] Prepare presentation/demo materials.
- [ ] Submit final report and repository link.

**Definition of done:** The project is reproducible enough for grading, the report explains the evidence clearly, and the demo shows the end-to-end system.

---

## Critical Path Summary

If time becomes tight, complete these in order:

1. Clean dataset and stable train/validation/test split.
2. Teacher prompt and teacher-output quality checks.
3. BM25 and sentence-transformer baselines.
4. Zero-shot Qwen baseline on a manageable sample or full test set.
5. QLoRA smoke test.
6. Full fine-tuning or smaller-model fallback.
7. Fine-tuned inference for the selected model through vLLM or the documented Transformers/PEFT fallback.
8. Automatic evaluation.
9. Small manual rewrite evaluation.
10. Error analysis.
11. Final report and demo.

## Minimum Viable Submission

If compute or API limits become severe, the minimum defensible version is:

- Cleaned dataset and documented preprocessing.
- Teacher-generated instruction dataset with quality checks.
- BM25 and sentence-transformer baseline results.
- Zero-shot Qwen results.
- One completed QLoRA run, even if on a smaller subset or smaller model.
- Fine-tuned inference for the selected model using vLLM if compute allows, or the documented Transformers/PEFT fallback if Colab serving is unstable.
- Automatic evaluation and small manual evaluation.
- Honest error analysis explaining limitations and compute constraints.
- Simple demo notebook using the best available model output.
