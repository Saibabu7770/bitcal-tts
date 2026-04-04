# Related Work and Differentiation

This document defines the closest prior work and the novelty boundary for `EdgeCompute-Orch`.

## Scope of This Project

`EdgeCompute-Orch` focuses on:

- inference-time compute policy selection per request,
- joint quality + latency + energy optimization,
- single consumer GPU deployment constraints.

This project does **not** introduce new quantization kernels or a new multi-GPU serving scheduler.

## Closest Prior Work

## 1) Adaptive Decoding and Inference-Time Compute

### Learning Adaptive LLM Decoding (arXiv:2603.09065)
- Link: https://arxiv.org/abs/2603.09065
- Focus: learned adaptive decoding under compute/token budgets.
- Why different from us: our objective is wall-clock and energy budget compliance on consumer hardware, not token-budget-only optimization.

### AdaSD (arXiv:2512.11280)
- Link: https://arxiv.org/abs/2512.11280
- Focus: adaptive speculative decoding dynamics.
- Why different from us: we orchestrate across multiple decoding families and include explicit multi-objective budget control.

### A*-Decoding / OptScale (2025 inference-time scaling line)
- Links:
  - https://arxiv.org/abs/2505.13672
  - https://arxiv.org/abs/2506.22376
- Focus: token-efficient search and compute allocation.
- Why different from us: these primarily optimize quality under compute assumptions; our emphasis is deployment-time budget enforcement (latency + energy) on local hardware.

## 2) SLO-Aware LLM Serving

### AdaServe (arXiv:2501.12162)
- Link: https://arxiv.org/html/2501.12162v1
- Focus: SLO-customized speculative decoding.
- Why different from us: mostly SLO/goodput in serving, without explicit quality-energy-latency tri-objective policy on single consumer GPUs.

### AdaSpec (arXiv:2503.05096)
- Link: https://arxiv.org/abs/2503.05096
- Focus: SLO-aware speculative decoding strategy.
- Why different from us: speculative-stack optimization; we compare and orchestrate broader policy set under multi-objective budgets.

### SOLA / ThunderServe (MLSys 2025)
- Links:
  - https://proceedings.mlsys.org/paper_files/paper/2025/hash/bc82dbfbfa43232be85b8d9838f49c3e-Abstract-Conference.html
  - https://proceedings.mlsys.org/paper_files/paper/2025/hash/c2a0e26dd9ee7d57e92bb1c24b39659a-Abstract-Conference.html
- Focus: datacenter scheduling and SLO optimization.
- Why different from us: cluster-scale scheduling vs local single-GPU orchestration.

## 3) Energy-Aware Inference Systems

### DynamoLLM / GreenServ / DVFS-style works
- Representative links:
  - https://www.microsoft.com/en-us/research/publication/dynamollm-designing-llm-inference-clusters-for-performance-and-energy-efficiency/
  - https://arxiv.org/html/2601.17551
  - https://arxiv.org/abs/2408.05235
- Focus: energy optimization at system or cluster level.
- Why different from us: our contribution is request-level policy selection under local deployment constraints with quality in-loop.

## 4) Consumer GPU Benchmarking

### Private LLM Inference on Consumer Blackwell GPUs (arXiv:2601.09527)
- Link: https://arxiv.org/abs/2601.09527
- Focus: broad cost/performance benchmarking.
- Why different from us: static configuration characterization, not adaptive inference-time compute controller with budget compliance guarantees.

## Novelty Statement (Safe)

We claim novelty in the **combination** of:

1. request-level inference-time compute orchestration,
2. joint quality-latency-energy objective,
3. single consumer GPU deployment and reproducible evaluation.

We do not claim to be the first adaptive decoding method or the first SLO-aware serving system.

## Citation and Writing Guardrails

- Cite all overlapping adaptive/SLO/energy papers listed above.
- Avoid absolute "first" claims.
- Use bounded language: "in the setting of single consumer GPUs with explicit quality-latency-energy budgets..."
