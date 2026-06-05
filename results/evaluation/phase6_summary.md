# Phase 6 Automatic Evaluation Summary

| system | score_evaluated_count | mae | rmse | pearson | spearman | accuracy | macro_f1 | explanation_evaluated_count | rouge_l_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bm25 | 726 | 20.2836 | 25.7651 | 11.4711 | 13.296 | 76.8595 | 48.6736 | 0 | None |
| zero_shot_qwen | 725 | 20.2372 | 24.0261 | 56.294 | 59.0974 | 82.069 | 67.1034 | 726 | 12.6682 |
| finetuned_qwen | 50 | 6.16 | 9.265 | 82.7219 | 85.2074 | 90.0 | 69.475 | 50 | 31.597 |
