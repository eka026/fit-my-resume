# Phase 8 Error Analysis Summary

## Scope

This analysis uses saved project artifacts rather than rerunning generation. Score weak cases come from `results/all_methods_test_results.jsonl`; raw system score columns above 100 are normalized to a 0-100 scale before error ranking. Qualitative failure labels come from saved validation outputs with parse status and generated JSON.

## Weak Score Examples

| system | pair_id | teacher_score | pred_score | absolute_error |
| --- | --- | --- | --- | --- |
| bm25 | test_19867922_job_000301_weak_random | 88.0 | 3.9556 | 84.0444 |
| bm25 | test_14556869_job_000693_strong_hybrid | 85.0 | 9.1727 | 75.8273 |
| bm25 | test_39166680_job_000693_weak_random | 85.0 | 10.1387 | 74.8613 |
| bm25 | test_19867922_job_000460_medium_tfidf | 80.0 | 7.6829 | 72.3171 |
| bm25 | test_13855004_job_000715_strong_hybrid | 75.0 | 6.3519 | 68.6481 |
| bm25 | test_28359817_job_000460_weak_random | 75.0 | 7.276 | 67.724 |
| bm25 | test_24763208_job_000550_strong_hybrid | 5.0 | 71.8938 | 66.8938 |
| bm25 | test_13264154_job_000811_medium_tfidf | 5.0 | 68.6984 | 63.6984 |
| bm25 | test_35651876_job_000655_strong_hybrid | 18.0 | 81.2941 | 63.2941 |
| bm25 | test_42641525_job_000811_weak_random | 5.0 | 66.7975 | 61.7975 |
| sentence_transf | test_19867922_job_000301_weak_random | 88.0 | 11.9257 | 76.0743 |
| sentence_transf | test_36694627_job_000734_strong_hybrid | 10.0 | 68.4655 | 58.4655 |
| sentence_transf | test_77266989_job_000095_strong_hybrid | 15.0 | 68.753 | 53.753 |
| sentence_transf | test_14593060_job_000100_strong_hybrid | 10.0 | 62.8837 | 52.8837 |
| sentence_transf | test_20321582_job_000644_strong_hybrid | 25.0 | 76.3803 | 51.3803 |
| sentence_transf | test_15535920_job_000433_medium_tfidf | 12.0 | 62.3423 | 50.3423 |
| sentence_transf | test_15932017_job_000143_strong_hybrid | 5.0 | 55.0454 | 50.0454 |
| sentence_transf | test_22349169_job_000644_strong_hybrid | 25.0 | 74.8977 | 49.8977 |
| sentence_transf | test_21567392_job_000229_medium_tfidf | 5.0 | 54.3021 | 49.3021 |
| sentence_transf | test_36694627_job_000550_medium_tfidf | 10.0 | 58.9295 | 48.9295 |

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
