# Topic Decision (Aligned Direction)

## Final Selected Topic (GO)

**EdgeCompute-Orch: Inference-Time Compute Orchestration under Latency and Energy Budgets on Consumer GPUs**

## Why This Topic Is Chosen

1. Directly aligns with inference-time compute and AI systems requirements.
2. Fits available hardware (single 8 GB class GPU) with practical experiment scope.
3. Strong engineering relevance for deployment teams (quality vs latency vs energy decisions).
4. Supports a high-value open-source contribution (benchmark + controller + reproducibility artifacts).

## Verified Prior Work (and Gap)

### Adaptive decoding exists, but often token-budget focused
- **Learning Adaptive LLM Decoding** (arXiv:2603.09065)
  - Link: https://arxiv.org/abs/2603.09065
- **AdaSD** (arXiv:2512.11280)
  - Link: https://arxiv.org/abs/2512.11280

### SLO-aware serving exists, often centered on speculative decoding or cluster scheduling
- **AdaServe** (arXiv:2501.12162)
  - Link: https://arxiv.org/html/2501.12162v1
- **AdaSpec** (arXiv:2503.05096)
  - Link: https://arxiv.org/abs/2503.05096
- **SOLA / ThunderServe** (MLSys 2025)
  - Links:
    - https://proceedings.mlsys.org/paper_files/paper/2025/hash/bc82dbfbfa43232be85b8d9838f49c3e-Abstract-Conference.html
    - https://proceedings.mlsys.org/paper_files/paper/2025/hash/c2a0e26dd9ee7d57e92bb1c24b39659a-Abstract-Conference.html

### Energy-aware optimization exists, usually at datacenter/cluster level
- **DynamoLLM / GreenServ / DVFS line**
  - Links:
    - https://www.microsoft.com/en-us/research/publication/dynamollm-designing-llm-inference-clusters-for-performance-and-energy-efficiency/
    - https://arxiv.org/html/2601.17551
    - https://arxiv.org/abs/2408.05235

### Consumer GPU benchmarks exist, but mostly static configuration studies
- **Private LLM Inference on Consumer Blackwell GPUs** (arXiv:2601.09527)
  - Link: https://arxiv.org/abs/2601.09527

## Novelty Boundary (Explicit)

This project does **not** claim:
- a new quantization algorithm,
- a new speculative decoding kernel,
- a new cluster scheduler.

This project **does** claim:
- per-request inference-time compute orchestration on consumer GPUs,
- explicit joint optimization of quality, wall-clock latency, and energy,
- reproducible open-source methodology for local deployment.

## Risk Assessment

- **Plagiarism risk:** low if writing is original and citations are complete.
- **Duplication risk:** moderate but defensible with strict scope and clear ablations.
- **Publication viability:** good for inference systems / applied ML systems venues and workshops.

## Decision

**GO**, with strict adherence to the novelty boundary and measurable budget-compliance claims.
