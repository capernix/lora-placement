# Phase 4 v2 follow-up runs on Kaggle

These scripts are additive wrappers around the existing Phase 4 v2 runner. They preserve the frozen RoBERTa commit, SST-2 fingerprint, preprocessing, split, optimizer, schedule, precision, five epochs, batch size 32, validation-only evaluation, and result schema. Layer indices are zero-based and LoRA is injected into `query` and `value` in the selected encoder blocks.

Run once in each notebook from the repository root:

```bash
pip install -r requirements-roberta-kaggle.txt
python3 run_phase4_v2.py prepare --config phase4_v2_config.json
python3 run_phase4_v2.py preflight --config phase4_v2_config.json --output-dir phase4_v2_preflight
```

The scripts refuse to train unless the split has fingerprint `e0f6962d602757cb` and manifest hash `bb3db736dead8440aa74c4863e61a1646ddf4f5fc3c7c5d5da60813ad70a3fae`. Every run asserts 73,728 LoRA adapter parameters and 665,858 total trainable parameters, including the classifier. It prints resolved target module paths and writes environment metadata into `result.json`.

## Team Member 1

Verify reproducibility first; this is a separate output directory and does not overwrite the completed reference:

```bash
python3 phase4_followup/scripts/run_followup.py verify \
  --split phase4_v2_preflight/phase4_v2_split.json \
  --artifacts-dir phase4_v2_preflight \
  --output-dir results/phase4_followup_repro
```

The command prints reproduced accuracy, stored reference accuracy, absolute difference, and warns above 0.002. It saves `reproducibility_report.json` and environment metadata. Run the assigned jobs:

```bash
python3 phase4_followup/scripts/build_followup_manifest.py --config spread_k3_r8 \
  --split phase4_v2_preflight/phase4_v2_split.json \
  --output phase4_v2_preflight/spread_k3_r8_manifest.json
python3 phase4_followup/scripts/run_team_member_1.py run --config spread_k3_r8 --seeds 0 1 2 \
  --split phase4_v2_preflight/phase4_v2_split.json \
  --manifest phase4_v2_preflight/spread_k3_r8_manifest.json \
  --output-dir results/phase4_followup
python3 phase4_followup/scripts/validate_result.py --input-dir results/phase4_followup \
  --manifest phase4_v2_preflight/spread_k3_r8_manifest.json
tar -czf team_member_1_phase4_followup.tgz results/phase4_followup \
  phase4_v2_preflight/spread_k3_r8_manifest.json
```

Expected assigned runs: 3.

## Team Member 2

Verify first with the same command above, then run:

```bash
python3 phase4_followup/scripts/build_followup_manifest.py --config strided_k6_r4 \
  --split phase4_v2_preflight/phase4_v2_split.json \
  --output phase4_v2_preflight/strided_k6_r4_manifest.json
python3 phase4_followup/scripts/run_team_member_2.py run --config strided_k6_r4 --seeds 0 1 2 \
  --split phase4_v2_preflight/phase4_v2_split.json \
  --manifest phase4_v2_preflight/strided_k6_r4_manifest.json \
  --output-dir results/phase4_followup
python3 phase4_followup/scripts/validate_result.py --input-dir results/phase4_followup \
  --manifest phase4_v2_preflight/strided_k6_r4_manifest.json
tar -czf team_member_2_phase4_followup.tgz results/phase4_followup \
  phase4_v2_preflight/strided_k6_r4_manifest.json
```

Expected assigned runs: 3.

## Team Member 3

Verify first with the same command above, then run both existing configurations with seeds 3 and 4:

```bash
python3 phase4_followup/scripts/build_followup_manifest.py --config extra_seeds \
  --split phase4_v2_preflight/phase4_v2_split.json \
  --output phase4_v2_preflight/extra_seeds_manifest.json
python3 phase4_followup/scripts/run_team_member_3.py run --config early3_r8 utility_top3_r8 --seeds 3 4 \
  --split phase4_v2_preflight/phase4_v2_split.json \
  --manifest phase4_v2_preflight/extra_seeds_manifest.json \
  --output-dir results/phase4_followup
python3 phase4_followup/scripts/validate_result.py --input-dir results/phase4_followup \
  --manifest phase4_v2_preflight/extra_seeds_manifest.json
tar -czf team_member_3_phase4_followup.tgz results/phase4_followup \
  phase4_v2_preflight/extra_seeds_manifest.json
```

Expected assigned runs: 4.

## Merge and analysis

Extract each archive into a separate incoming directory, then validate and merge with its matching manifest. The merge refuses duplicate run IDs and never overwrites a completed run unless `--force` is explicit:

```bash
python3 phase4_followup/scripts/merge_results.py --incoming results/phase4_followup_from_member_1 \
  --manifest phase4_v2_preflight/spread_k3_r8_manifest.json --destination phase4-18runs
python3 phase4_followup/scripts/merge_results.py --incoming results/phase4_followup_from_member_2 \
  --manifest phase4_v2_preflight/strided_k6_r4_manifest.json --destination phase4-18runs
python3 phase4_followup/scripts/merge_results.py --incoming results/phase4_followup_from_member_3 \
  --manifest phase4_v2_preflight/extra_seeds_manifest.json --destination phase4-18runs
```

Then produce the summary table and paired differences:

```bash
python3 phase4_followup/scripts/analyze.py --roots phase4-18runs analysis/phase4/results/phase4_v2 results/phase4_followup \
  --output-dir analysis/phase4_followup
```

The analysis reports raw seed accuracies, standard deviations, mean LoRA utility, and paired differences only for matching seed IDs. Interpretations are scoped to RoBERTa-base + SST-2 + the current Phase 4 protocol and are descriptive, not universal claims.
