# Private upload readiness and publication holds

This candidate is prepared for a **private** review repository. It is not approved for public release or for an unqualified claim of end-to-end reproduction.

- `docs/publication_holds.csv` lists the original 12 path-bearing files, the packaging helper, and baseline source files excluded from the initial commit pending authorship/redistribution review. No original archive was changed.
- The held `run_reward_ablation.py` reconstructs and patches a work copy before packaging results; it is not the standalone five-run exact-comparison entry. The held `check_main_integration.py` copies source and rewrites `init3.py`; a path substitution alone would not make that reconstruction safe. Existing `run_exact_s1.py` and `check_actual_schedules.py` snapshots remain, and `check_actual_schedules_portable.py` supplies a path-only variant for the included schedule snapshot.
- Historical nine-instance input, Q-table, HV/IGD reference, and paper-figure mappings remain unconfirmed. The recorded R1-5 ablation code differs from the final revised decoder and objective implementation.
- The full raw logs remain in original archives, indexed by `docs/large_data_manifest.csv`. No unverified download link or full-reproduction assertion is supplied.
- A clean-environment run and source-license review are required before a public release. No experiment was run during this packaging and upload-preparation step.
