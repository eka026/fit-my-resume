# Phase 9 Notebook Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a notebook demo that shows the complete FitMyResume resume-job matching workflow without requiring local GPU serving.

**Architecture:** The notebook uses existing checked-in fine-tuned validation outputs as the default presentation path, with optional cells documenting live vLLM and Transformers/PEFT inference. It includes local helper functions for JSON loading, output parsing, input length checks, and display formatting so the demo is self-contained.

**Tech Stack:** Jupyter notebook JSON, Python standard library, existing project JSONL artifacts.

---

### Task 1: Create Notebook Demo

**Files:**
- Create: `notebooks/fitmyresume_demo.ipynb`

- [ ] **Step 1: Create the notebook with markdown overview cells**

Add title, goal, backend note, and run instructions. Explain that the default path uses `results/finetuned_qwen_validation_transformers_sample50_outputs.jsonl` and does not require GPU.

- [ ] **Step 2: Add helper cells**

Add Python helpers for project-root discovery, JSONL loading, instruction input parsing, parsed-output normalization, length checks, and readable display of score, explanation, and resume suggestions.

- [ ] **Step 3: Add polished example cells**

Load one parseable fine-tuned output, show the original resume and job description, validate length, then display the model output sections.

- [ ] **Step 4: Add optional live inference cells**

Document commands for vLLM and Transformers/PEFT live inference, keeping them disabled/commented so the demo runs on a CPU notebook.

### Task 2: Update Project Docs and Task List

**Files:**
- Modify: `README.md`
- Modify: `docs/Chronological Project Tasks.md`

- [ ] **Step 1: Add README demo instructions**

Add a `Notebook Demo` section with the notebook path, default artifact, and optional live inference note.

- [ ] **Step 2: Mark Phase 9 complete for notebook scope**

Update Phase 9 checkboxes to reflect the notebook demo decision, saved-output fallback, notebook input/output sections, parsing/error handling, length checks, tested examples, polished presentation example, and saved notes.

### Task 3: Validate

**Files:**
- Read: `notebooks/fitmyresume_demo.ipynb`
- Read: `results/finetuned_qwen_validation_transformers_sample50_outputs.jsonl`

- [ ] **Step 1: Validate notebook JSON**

Run a Python JSON load on `notebooks/fitmyresume_demo.ipynb`. Expected: no JSON parse errors.

- [ ] **Step 2: Validate helper logic against existing artifact**

Run a small Python script that loads the saved artifact, finds parseable outputs, extracts input sections, and confirms score/explanation/suggestion fields exist. Expected: at least one valid demo example.

- [ ] **Step 3: Run focused tests**

Run existing tests for inference parsing helpers: `pytest tests/test_finetuned_vllm_inference.py tests/test_finetuned_transformers_inference.py -q`. Expected: all pass.
