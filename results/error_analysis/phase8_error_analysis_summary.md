# Phase 8 Error Analysis Summary

## Scope

This analysis uses saved project artifacts rather than rerunning generation. Score weak cases come from `results/all_methods_test_results.jsonl`; qualitative failure labels come from saved validation outputs with parse status and generated JSON.

## Weak Score Examples

| system | pair_id | teacher_score | pred_score | absolute_error |
| --- | --- | --- | --- | --- |
| bm25 | test_27058381_job_000755_strong_hybrid | 55.0 | 1044.2708227093124 | 989.2708 |
| bm25 | test_72652441_job_000655_strong_hybrid | 25.0 | 902.5378925386377 | 877.5379 |
| bm25 | test_20748929_job_000655_strong_hybrid | 45.0 | 917.0226396414595 | 872.0226 |
| bm25 | test_15620421_job_000655_strong_hybrid | 25.0 | 875.2289713868504 | 850.229 |
| bm25 | test_59696315_job_000811_strong_hybrid | 40.0 | 877.146675001745 | 837.1467 |
| bm25 | test_12858898_job_000115_strong_hybrid | 55.0 | 890.7105478717802 | 835.7105 |
| bm25 | test_35651876_job_000655_strong_hybrid | 18.0 | 848.930245013506 | 830.9302 |
| bm25 | test_25543217_job_000115_strong_hybrid | 25.0 | 853.1420864101532 | 828.1421 |
| bm25 | test_34131484_job_000655_strong_hybrid | 25.0 | 830.5767400502126 | 805.5767 |
| bm25 | test_12230301_job_000031_strong_hybrid | 55.0 | 857.4861340654726 | 802.4861 |
| sentence_transf | test_19867922_job_000301_weak_random | 88.0 | 11.925704002380371 | 76.0743 |
| sentence_transf | test_36694627_job_000734_strong_hybrid | 10.0 | 68.4654541015625 | 58.4655 |
| sentence_transf | test_77266989_job_000095_strong_hybrid | 15.0 | 68.75303649902344 | 53.753 |
| sentence_transf | test_14593060_job_000100_strong_hybrid | 10.0 | 62.88369369506836 | 52.8837 |
| sentence_transf | test_20321582_job_000644_strong_hybrid | 25.0 | 76.38031768798828 | 51.3803 |
| sentence_transf | test_15535920_job_000433_medium_tfidf | 12.0 | 62.34225082397461 | 50.3423 |
| sentence_transf | test_15932017_job_000143_strong_hybrid | 5.0 | 55.045413970947266 | 50.0454 |
| sentence_transf | test_22349169_job_000644_strong_hybrid | 25.0 | 74.89769744873047 | 49.8977 |
| sentence_transf | test_21567392_job_000229_medium_tfidf | 5.0 | 54.30213928222656 | 49.3021 |
| sentence_transf | test_36694627_job_000550_medium_tfidf | 10.0 | 58.92951583862305 | 48.9295 |

## Failure Mode Counts

| failure_mode | count |
| --- | --- |
| Invalid JSON | 25 |
| Wrong or poorly calibrated score | 1 |
| Missing important job requirements | 0 |
| Overly generic explanation | 2 |
| Fabricated resume content risk | 25 |
| Rewrite that removes important candidate evidence | 0 |
| Prompt injection vulnerability | 0 |

## Likely Failure Sources

| source | count |
| --- | --- |
| Compute-constrained truncation | 25 |
| Dataset noise | 25 |
| Model limitations | 3 |
| Prompt design | 52 |
| Teacher-output quality | 1 |

## Iteration Notes

- No second fine-tuning run was performed in this phase because the current artifacts are sufficient for the CS455 final report and additional training would require GPU time.
- Targeted prompt iteration should focus on stricter JSON-only generation, requirement-by-requirement evidence, and requiring every resume suggestion to cite resume evidence or use `add_if_true`.
- Low-quality teacher examples should be removed when the teacher score conflicts with clear resume/job evidence or when suggestions lack evidence.
