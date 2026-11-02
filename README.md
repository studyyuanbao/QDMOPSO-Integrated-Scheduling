# A dynamic evolutionary state feedback-driven hyper-heuristic method for integrated procurement and multi-stage assembly scheduling under the industrial internet

Code, instance inputs, experiment protocols, numerical results, and model-validation records for the Engineering Optimization revision. The six experiment modules preserve their recorded code versions. The repository is a review candidate; clean-environment reproduction has not been verified.

| Module | Code and entry | Inputs and protocol | Results |
| --- | --- | --- | --- |
| [01 Fixed strategies](experiments/01_fixed_strategies/) | `code_candidates/compare_all.py` | [Nine-instance candidate inputs](instances/original_9_diagnostic_candidates/) | Formal historical run mapping has not been established. |
| [02 Baseline comparison](experiments/02_baseline_comparison/) | `code_candidates/C8-5.py` | Nine-instance candidate inputs | Formal result and historical HV/IGD reference mapping has not been established. |
| [03 Module ablation](experiments/03_module_ablation/) | `run_ablation.py`, `code/` | S1/M2/L3, `protocol.json` | [Run summaries and selected per-run records](results/03_module_ablation/). |
| [04 Parameter sensitivity](experiments/04_parameter_sensitivity/) | `run_job.py`, `code/` | M1, `jobs.json`, `protocol.json` | [Candidate summaries and selected per-run records](results/04_parameter_sensitivity/). |
| [05 Equal-budget comparison](experiments/05_equal_budget_comparison/) | `run_job.py`, `code/` | Six instances, `jobs.json`, `protocol.json` | [Formal-test summaries and selected per-run records](results/05_equal_budget_comparison/). Training is recorded separately. |
| [06 Exact comparison](experiments/06_exact_comparison/) | `run_exact_s1_portable.py`, `run_epsilon_s1_portable.py` | S1, `algorithm-run-manifest.json` | [Exact comparison and five linked QDMOPSO A runs](results/06_exact_comparison/). |
| [Validation](validation/) | Gurobi formulation check and fixed-schedule checker | Constructed cases and recorded schedules | [Validation outputs](validation/). |

The recorded R1-5 ablation uses its own decoder and cost implementation; its results should be interpreted for that recorded version. The S1 exact-comparison A runs are separate from the six-instance formal tests. Source files for four baseline methods remain excluded because redistribution permission has not been established. Some modules therefore cannot be run from a clean clone.

See the [experiment index](docs/experiment_index.md), [run guide](docs/run_guide.md), [metric definitions](docs/metrics.md), and [selected evidence manifest](docs/selected_evidence_manifest.csv). The run commands are archival entry points, not a claim that all historical paper results have been reproduced. Gurobi runs require a valid license. No license for reuse of this repository is granted by its presence here.
