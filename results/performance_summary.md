# Performance Summary

## Recorded Results

| Implementation | Configuration | Execution Time (s) | Verification |
|---|---|---:|---:|
| Sequential | 1 CPU execution | 204.398481 | 4000.00 |
| OpenMP | 8 threads | 43.578664 | 4000.00 |
| MPI | 4 processes | 60.802443 | 4000.00 |
| CUDA | 16 × 16 threads/block, total phase | 0.080268 | 4000.00 |

CUDA kernel time: `0.067203 s`.

## Recorded Speedup

Relative to the sequential result:

- OpenMP: `4.69×`
- MPI: `3.36×`
- CUDA total phase: `2546.45×`

These are recorded experimental comparisons. Timing is environment-dependent and the timing scopes are not identical across all execution models.
