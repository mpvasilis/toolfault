# ToolFault experiment audit

Published grid contains 4,992 episodes, including 4 endpoint errors.

## Silent faults in the published model runs

The percentile intervals resample task IDs 2,000 times and preserve the fault episodes within each task. They quantify variation across tasks in the published world. AUROC measures answer correctness among valid confidence submissions. Review and unsafe rates use all scorable silent episodes.

| Agent | Fault fired | Valid submissions | AUROC [95% interval] | Review, no gate → ≥0.95 | Unsafe, no gate → ≥0.95 |
|---|---|---|---|---|---|
| deepseek-v3.1 | 109/112 | 32 | n/a | 71.4% → 84.8% | 5.4% → 3.6% |
| gpt-oss-120b | 83/112 | 52 | 0.234 [0.101, 0.363] | 51.8% → 67.0% | 12.5% → 10.7% |
| gpt-oss-20b | 75/112 | 72 | 0.636 [0.500, 0.758] | 34.8% → 44.6% | 20.5% → 16.1% |
| kimi-k2-0905 | 90/110 | 70 | 0.448 [0.262, 0.578] | 36.4% → 50.0% | 13.6% → 9.1% |
| naive | 61/112 | 112 | 0.500 [0.500, 0.500] | 0.0% → 100.0% | 51.8% → 0.0% |
| oracle | 92/112 | 85 | n/a | 24.1% → 68.8% | 0.0% → 0.0% |
| qwen3-235b-a22b | 100/112 | 45 | 0.365 [0.144, 0.601] | 56.2% → 60.7% | 8.0% → 8.0% |
| secondary-only | 63/112 | 112 | 0.500 [0.500, 0.500] | 0.0% → 100.0% | 53.6% → 0.0% |

The threshold is a retrospective illustration. Review rates assume any diverted submission goes to a reviewer. The audit does not measure human review time or quality.

## Scripted policies across six worlds

| World seed | Agent | Resolved, all | Resolved, silent | Unsafe, silent |
|---|---|---|---|---|
| 7 | oracle | 624/624 | 100.0% | 0.0% |
| 7 | naive | 308/624 | 48.2% | 51.8% |
| 7 | secondary-only | 330/624 | 46.4% | 53.6% |
| 13 | oracle | 621/621 | 100.0% | 0.0% |
| 13 | naive | 293/621 | 44.0% | 56.0% |
| 13 | secondary-only | 329/621 | 44.0% | 56.0% |
| 17 | oracle | 621/621 | 100.0% | 0.0% |
| 17 | naive | 307/621 | 46.8% | 53.2% |
| 17 | secondary-only | 340/621 | 47.7% | 52.3% |
| 41 | oracle | 622/622 | 100.0% | 0.0% |
| 41 | naive | 288/622 | 48.2% | 51.8% |
| 41 | secondary-only | 334/622 | 46.4% | 53.6% |
| 53 | oracle | 623/623 | 100.0% | 0.0% |
| 53 | naive | 304/623 | 48.6% | 51.4% |
| 53 | secondary-only | 329/623 | 45.9% | 54.1% |
| 79 | oracle | 622/622 | 100.0% | 0.0% |
| 79 | naive | 322/622 | 49.1% | 50.9% |
| 79 | secondary-only | 345/622 | 48.2% | 51.8% |
