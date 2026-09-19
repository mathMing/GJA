# Corrected feasibility experiment

Use the corrected entry points below. Older run_all.py outputs are historical and must not be used as validated evidence.

```powershell
cd X:\GJA\WORK
& D:\conda\envs\pytorch\python.exe src\ars_experiment.py --stage test
& D:\conda\envs\pytorch\python.exe src\ars_experiment.py --stage all
& D:\conda\envs\pytorch\python.exe src\ars_report.py
```

Outputs: results/ars_validated_v1. Formal run: 5 seeds, 5 folds, 4 scorers, 600 synthetic repetitions of 20,000 samples. A new run overwrites this output directory's generated files; original source workbook is read-only.

Main report: advisor_summary.md. Detailed limitations: experiment_report.md. Patient-level outputs contain local sensitive identifiers and should remain local. The six-section HTML is a printable draft, not a visually verified PDF/PPT.

Do not call these results clinical certification, independent external validation, or proof of acquisition benefit. No-service risk is undefined. CV repeats are not new patients.
