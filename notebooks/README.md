# Notebooks

This directory contains interactive GPU validation workflows, not submission
artifacts. The notebook calls the benchmark and accuracy tools from the
repository instead of maintaining duplicate implementations.

- `colab_benchmark.ipynb`: Colab T4 functional, stability, workload, and
  accuracy validation for the remaining submission process. It clones
  `Platypus27-coder/viettel-ai-race-llm-serving` into the Colab runtime, so no
  folder upload is required.

[Open `colab_benchmark.ipynb` in Colab](https://colab.research.google.com/github/Platypus27-coder/viettel-ai-race-llm-serving/blob/main/notebooks/colab_benchmark.ipynb)

T4 latency results must not be used to predict H200 FP8 performance.
