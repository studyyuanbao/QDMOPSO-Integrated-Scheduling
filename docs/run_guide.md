# Run guide

The archived newer-experiment environment reports Python 3.13.5. See `requirements.txt` for observed Python dependencies. Historical 01/02 Python versions are unconfirmed. Exact solving and mathematical validation require Gurobi 13.0.3 and a valid license. Commands below describe entry syntax only; they were not executed or tested in a clean environment while preparing this repository. Run them in a separate working copy so archived records remain intact.

| Module | Entry from repository root | Inputs | Recorded results / limitation |
| --- | --- | --- | --- |
| 01 | `python experiments/01_fixed_strategies/code_candidates/compare_all.py` | Candidate nine-instance inputs; historical Q table not verified | Formal output correspondence unconfirmed. |
| 02 | `python experiments/02_baseline_comparison/code_candidates/C8-5.py` | Candidate nine-instance inputs | Baseline source files excluded pending redistribution rights; clean-clone run incomplete. |
| 03 | `python experiments/03_module_ablation/run_ablation.py --case S1` | `S1/config.py`, `S1/input.json`, `protocol.json`; substitute M2/L3 | `results/03_module_ablation/`; original decoder and objective are preserved. |
| 04 | `python experiments/04_parameter_sensitivity/run_job.py --job 1` | `jobs.json`, `protocol.json`, `M1-config.py` | `results/04_parameter_sensitivity/`; ordered jobs include training dependencies; baseline source rights unresolved. |
| 05 | `python experiments/05_equal_budget_comparison/run_job.py --job 1` | `jobs.json`, `protocol.json`, `instances/` | `results/05_equal_budget_comparison/`; ordered jobs include training dependencies; baseline source rights unresolved. |
| 06 | `python experiments/06_exact_comparison/run_exact_s1_portable.py` then `python experiments/06_exact_comparison/run_epsilon_s1_portable.py` | `S1-input.json` | `results/06_exact_comparison/`; licensed Gurobi required. |
| Validation | `python validation/formulation_checks/validate_full_model.py`; `python validation/decoder_consistency/check_actual_schedules_portable.py` | Constructed cases; `main-schedules.json` | `validation/` recorded outputs; licensed Gurobi required. |

The portable entry copies only adjust local paths. Archived result files were not recomputed. `q-final.pkl` checkpoints are pickle data and should be opened only from trusted sources.
