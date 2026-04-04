# Project Plan: BitCal-TTS

End-to-end execution plan for:
**Bit-Calibrated Test-Time Scaling for Quantized Reasoning Models**

## 1) Objective

Build and validate a lightweight, bit-aware calibration framework that improves adaptive halting decisions for quantized reasoning LLMs under fixed inference budgets, without retraining base models.

## 2) Success Criteria

Project is considered successful if all conditions below are met:

- Improves budgeted reasoning accuracy over fixed-budget and non-bit-aware halting baselines.
- Reduces halting errors (premature-stop and over-think rates) in quantized settings.
- Improves quality-efficiency Pareto (quality vs reasoning tokens/latency).
- Reproducible results with fixed seeds and end-to-end scripts.
- Open-source artifact is complete and runnable on target hardware.

## 3) Scope

### In Scope
- Quantized inference of open reasoning models.
- Signal extraction (entropy, hidden-state consistency, trace stability).
- Bit-aware confidence correction and adaptive halting policy.
- Evaluation on math, science, and code reasoning tasks.

### Out of Scope
- Training a new base LLM.
- Proposing a new quantization algorithm.
- Building custom inference kernels.

## 4) Workstreams

1. **Infrastructure and Reproducibility**
2. **Baselines and Evaluation Pipeline**
3. **Signal Engineering**
4. **Bit-Aware Calibration**
5. **Adaptive Policy and Ablations**
6. **Paper + Open-Source Release**

## 5) Phase Plan (Milestones)

## Phase A: Foundation (Week 1)

### Tasks
- Finalize hypotheses and metric definitions.
- Freeze model/task shortlist that fits local GPU constraints.
- Set up project skeleton and dependency lock.
- Create run config templates and logging format.

### Deliverables
- `configs/` first version
- baseline run script skeleton
- metric definition sheet

## Phase B: Baseline System (Week 2)

### Tasks
- Implement fixed-budget inference runner.
- Add dataset loaders and prompt templates.
- Validate deterministic logging and output parsing.

### Deliverables
- baseline result tables
- latency/token usage logs
- sanity-check report

## Phase C: Signal Pipeline (Week 3)

### Tasks
- Implement entropy extraction per step.
- Implement hidden-state consistency signals.
- Implement reasoning-trace stability measures.

### Deliverables
- `src/signals/` functional pipeline
- signal quality diagnostics

## Phase D: BitCal Module (Week 4)

### Tasks
- Fit bit-aware confidence correction on calibration split.
- Compare calibration before/after correction.
- Validate no base-model retraining requirement.

### Deliverables
- `src/calibrator/` module
- calibration performance report

## Phase E: Adaptive Halting Policy (Week 5)

### Tasks
- Implement continue/stop/escalate policy.
- Integrate budget controller (token and optional latency bound).
- Add fallback behavior for uncertain cases.

### Deliverables
- `src/policy/` policy implementation
- first adaptive-vs-static comparison

## Phase F: Ablations + Robustness (Week 6)

### Tasks
- Signal ablations (remove one signal at a time).
- Quantization setting sweep (weight/activation/KV where supported).
- OOD and hard-instance testing.

### Deliverables
- ablation tables
- robustness report

## Phase G: Final Experiments (Week 7)

### Tasks
- Rerun key experiments with fixed seeds.
- Collect confidence intervals and error bars.
- Produce final plots and summary tables.

### Deliverables
- camera-ready result assets
- final metrics bundle

## Phase H: Paper + Release (Week 8)

### Tasks
- Write methods/results/related work/failure analysis.
- Prepare open-source package and replication guide.
- Final consistency check between paper claims and code evidence.

### Deliverables
- paper draft
- public repo-ready package
- reproducibility checklist completed

## 6) Weekly Timeline (Compact)

- **Week 1:** setup + scope freeze  
- **Week 2:** baseline pipeline  
- **Week 3:** signal extraction  
- **Week 4:** bit-aware calibration  
- **Week 5:** adaptive halting policy  
- **Week 6:** ablations + robustness  
- **Week 7:** final experiments  
- **Week 8:** paper + open-source release

## 7) Experiment Matrix

- Models: 2-3 quantized reasoning-capable open models
- Tasks: math / science / code reasoning benchmarks
- Budgets: low / medium / high reasoning-token budgets
- Quantization modes:
  - weight quantization levels
  - activation quantization where supported
  - KV-cache quantization where supported
- Policies:
  - fixed-budget baseline
  - non-bit-aware adaptive halting
  - BitCal-TTS

## 8) Core Metrics

- Task accuracy / pass@k
- Accuracy at fixed token budget
- Premature-stop rate
- Over-think rate
- Avg reasoning tokens
- p95 latency (optional if stable)
- Pareto frontier: quality vs token/latency budget

## 9) Risks and Mitigations

### Risk: Weak improvement over baseline
- Mitigation: strengthen calibration split design, add ensemble-free signal fusion, broaden ablations.

### Risk: Hardware limits on model/task scale
- Mitigation: prioritize fewer models with stronger benchmark depth; use shorter controlled subsets early.

### Risk: Signal noise in quantized runs
- Mitigation: smooth with rolling statistics; verify with repeated runs and seed control.

### Risk: Overlap concerns in review
- Mitigation: keep claims narrow (bit-aware halting calibration under quantized budgets), cite closely related work explicitly.

## 10) Open-Source Release Checklist

- Source code for runner, signals, calibrator, policy, eval
- Configs and exact command examples
- Raw logs and processed result tables
- Plot generation scripts
- Environment file (`requirements.txt` or lock file)
- Reproduction guide with expected runtime and hardware notes

## 11) Immediate Next 3 Actions

1. Freeze model/task shortlist and benchmark subsets.
2. Implement fixed-budget baseline runner with full logging.
3. Start signal extraction module and validate on 20-50 sample runs.
