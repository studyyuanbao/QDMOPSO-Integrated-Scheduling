# Code versions and evidence limits

Each experiment contains its own source snapshot. Identical filenames do not establish identical code. The original archives and their SHA256/protocol manifests remain the version evidence; see `data_manifest.csv` for copied-file hashes.

| Module | Snapshot | Status |
| --- | --- | --- |
| 01 fixed strategies | `experiments/01_fixed_strategies/code_candidates/` | Candidate scripts; Q-table content and historical per-instance execution linkage unconfirmed. |
| 02 five-algorithm comparison | `experiments/02_baseline_comparison/code_candidates/` | Candidate historical code; no complete nine-instance run-to-figure mapping. |
| 03 module ablation | `experiments/03_module_ablation/code/` | Actual R1-5 recorded version. Its `decoder.py` and `objective.py` are retained as used. |
| 04 M1 parameter study | `experiments/04_parameter_sensitivity/code/` | Tied to the frozen 35-configuration protocol and run summaries. |
| 05 six-instance study | `experiments/05_equal_budget_comparison/code/` | Tied to the 630 recorded training/test jobs and code hashes. |
| 06 exact comparison | `experiments/06_exact_comparison/qdmopso_A_runs/code/` | Group-A reward diagnostic snapshot linked by the exact comparison manifest; not the six-instance test code. |
| Validation | `validation/formulation_checks/type_revision/` and `validation/decoder_consistency/main-integration/` | Structural and fixed-schedule checks, not historical performance runs. |

The R1-5 decoder pools inventory by component name, can use instance identity for changeovers, and has a material-wait path that can overwrite an earlier start bound. The model-aligned decoder pools by product type and component identity, records the consumed physical units, and takes the maximum of predecessor completion, machine/setup availability, and material arrival. The R1-5 objective searches for a strictly later use time and falls back to `end + 10`; the model-aligned objective computes holding time from recorded unit consumption, including zero waiting. These differences may alter schedules and costs. The R1-5 results therefore describe their recorded version and do not independently establish module contributions under the final model-aligned implementation.
