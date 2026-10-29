# cuda-rapids

Learning NVIDIA RAPIDS one notebook at a time. Every notebook is self-contained (setup, data, timing, plots inline).

**Target hardware:** NVIDIA T4 (16 GB, Turing / sm_75). AMD/ROCm variants are out of scope for now.

```
rapids/
├── 01_cuDF/       # GPU DataFrames, cudf.pandas
├── 02_cuVS/       # Vector search: brute force, IVF, CAGRA
├── 03_cuGraph/    # Graph analytics, nx-cugraph
├── 04_cuML/       # GPU machine learning, cuml.accel
├── 05_RMM_Dask/   # Memory pools, Dask-cuDF
└── 06_cuCIM/      # GPU image processing
```

## Benchmark conventions
- Warmup runs, then the median of several repeats
- GPU synchronized before each timer stops
- Host-to-device transfer time reported separately from compute
- GPU model and library versions logged in the first cell
