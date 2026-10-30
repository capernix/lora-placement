"""Build and validate the frozen 66-pair screening manifest."""
from __future__ import annotations
from phase4_v2_io import sha256_payload
from phase4_v2_pairs import canonical_pair_id, enumerate_pairs, permute_pairs

EXPECTED_PARAMS = 665858

def build_pair_manifest(split_manifest_hash, protocol_version="phase4_v2_pairs", permutation_seed=20261036):
    pairs = enumerate_pairs()
    execution = permute_pairs(pairs, permutation_seed)
    entries = []
    for position, pair in enumerate(execution):
        entries.append({"pair_id": canonical_pair_id(pair), "layers": list(pair), "rank": 12, "alpha": 24,
                        "training_seed": 0, "permutation_seed": permutation_seed, "execution_position": position,
                        "protocol_version": protocol_version, "split_manifest_sha256": split_manifest_hash,
                        "expected_trainable_parameters": EXPECTED_PARAMS})
    runs = [{"run_id": f"{entry['pair_id']}_seed{entry['training_seed']}",
             "configuration_id": entry["pair_id"], "layers": entry["layers"],
             "rank": entry["rank"], "alpha": entry["alpha"], "seed": entry["training_seed"],
             "expected_trainable_parameters": EXPECTED_PARAMS} for entry in entries]
    manifest = {"protocol_version": protocol_version, "manifest_type": "pairwise_screening", "pair_count": len(entries),
                "split_manifest_sha256": split_manifest_hash, "permutation_seed": permutation_seed,
                "pairs": entries, "run_count": len(runs), "runs": runs}
    manifest["manifest_sha256"] = sha256_payload(manifest)
    return manifest
