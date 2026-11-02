# Metric scopes

## Original nine-instance studies

The candidate `C8-5.py` builds an IGD reference set from the nondominated union of collected output archives and an HV reference point `(1.5 × max Time + 1, 1.5 × max Cost + 1)`. It computes HV area and IGD Euclidean distance on raw Time/Cost values. **The script and the nine paper results have not been connected by a complete run manifest.** Nine instance-specific reference coordinates, saved reference fronts, successful run counts, and any historical normalization remain unconfirmed. `compare_all.py` is a different comparison and cannot establish the five-algorithm paper protocol. Existing CSVs under `results/02_baseline_comparison/historical_candidates/` are unassigned candidates.

## M1 parameter study

`experiments/04_parameter_sensitivity/protocol.json` and `analysis-reference.json` define a fixed-reference normalized output-archive HV for the 35 candidate configurations. Three same-seed tests per configuration were used for the reported means; QDMOPSO has one 20-episode training sequence per configuration. These values apply to M1 only.

## Six-instance equal-budget study

`results/05_equal_budget_comparison/analysis_package/analysis/metric-protocol.json` is authoritative for this study. Normalized HV is the dominated area divided by the product of the fixed Time and Cost reference coordinates for that instance. Empirical IGD uses Euclidean distances after dividing each objective by those reference coordinates; its reference front is the nondominated union of formal test output fronts for that instance. The fixed references and empirical fronts are in the matching analysis package. These definitions must not be imputed to the older nine-instance figures.

## S1 exact comparison

The five QDMOPSO runs referenced by `algorithm-run-manifest.json` belong to the group-A reward diagnostic, with 100 generations and varying actual objective-evaluation counts. Their metric reference is recorded in that manifest. They are distinct from the six-instance formal tests.
