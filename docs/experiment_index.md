# Experiment index

Paths are relative to the repository. The listed entry points and records identify archived material; historical identity is stated only where supported by a run manifest.

| Module; paper location | Entry and input | Protocol | Recorded output and linkage |
| --- | --- | --- | --- |
| 01; Section 5.1, Table 7–8, Figs. 7–8 | `experiments/01_fixed_strategies/code_candidates/compare_all.py`; `instances/original_9_diagnostic_candidates/` | Candidate script: 50 particles, 100 generations, 20 runs, seeds 42–61 | Formal table/figure run linkage and Q-table identity unconfirmed. |
| 02; Section 5.2, Figs. 9–14 | `experiments/02_baseline_comparison/code_candidates/C8-5.py`; candidate nine-instance inputs | Candidate script: 20 runs, seeds 32–51 | Formal result, HV-reference and IGD-front linkage unconfirmed; baseline source is excluded pending redistribution rights. |
| 03; Section 5.1 supplement, R1-5 | `experiments/03_module_ablation/run_ablation.py`; `S1/`, `M2/`, `L3/` | `protocol.json`; 5,000 evaluations; three training sequences and nine paired tests per variant | `results/03_module_ablation/`; recorded decoder/cost version differs from final revised model. |
| 04; Section 5, AE-3 | `experiments/04_parameter_sensitivity/run_job.py`; M1 input and `jobs.json` | `protocol.json`; 35 parameter configurations, 5,000 evaluations | `results/04_parameter_sensitivity/`; selected per-run data retained. |
| 05; Section 5.2.1 | `experiments/05_equal_budget_comparison/run_job.py`; six instances and `jobs.json` | `protocol.json`; 270 formal tests and 360 separate QDMOPSO training episodes, 5,000 evaluations each | `results/05_equal_budget_comparison/`; summaries and selected per-run records. |
| 06; Section 5, AE-2 | `experiments/06_exact_comparison/run_exact_s1_portable.py`, `run_epsilon_s1_portable.py`; `S1-input.json` | `algorithm-run-manifest.json`; Gurobi 13.0.3; QDMOPSO A seeds 44001–44005 | `results/06_exact_comparison/`; five A runs are training-mode records, separate from 05. |
| Validation; Section 3.2, R1-1–3 | `validation/formulation_checks/validate_full_model.py`; `validation/decoder_consistency/check_actual_schedules_portable.py` | Constructed cases and fixed schedule snapshot | `validation/` outputs; 130 fixed schedules are records, not independent optimization instances. |

File-level source hashes are in `docs/selected_evidence_manifest.csv`. Metric scopes are in `docs/metrics.md`; entry syntax and dependencies are in `docs/run_guide.md`.
