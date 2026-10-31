# Token Usage Analysis

## Experiment

- Baseline trajectory: `django__django-15368-baseline-trajectory.json`
- Compacted trajectory: `django__django-15368-trajectory.json`

## Results

| Metric | Baseline | Compacted |
|---|---:|---:|
| Agent steps | 38 | 20 |
| Compaction events | 0 | 9 |
| Agent prompt tokens | 888,098 | 88,893 |
| Compaction prompt tokens | 0 | 48,508 |
| Total prompt tokens | 888,098 | 137,401 |
| Total completion tokens | 9,729 | 14,553 |
| Total tokens | 897,827 | 151,954 |
| SWE-bench result | RESOLVED | RESOLVED |

## Token Savings

- Baseline total: **897,827 tokens**
- Compacted total: **151,954 tokens**
- Tokens saved: **745,873**
- Total token reduction: **83.08%**

## Compaction Analysis

- Compaction events: 9
- Estimated tokens before compaction (summed snapshots): 71,713
- Estimated tokens after compaction (summed snapshots): 37,088
- Snapshot reduction: **48.28%**

## Observations

1. The baseline used 38 agent steps.
2. The compacted run used 20 agent steps.
3. Context compaction occurred 9 times.
4. Compaction calls consumed 56,260 additional tokens.
5. Total token usage decreased by 83.08%.

## Conclusion

This experiment compares context compaction against a full-context baseline using the same SWE-bench instance. Token totals include both normal agent calls and compaction API calls. The recorded context estimates are summed across compaction snapshots and are not unique-token savings.

SWE-bench evaluation results are supplied separately because they are not stored in the trajectory JSON.
