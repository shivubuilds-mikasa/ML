"""Export the trained scikit-learn forest for client-side GitHub Pages inference.

Run from the repository root with: python website/export_model.py
The browser decompresses the packed tree data and evaluates the exact trees
from models/best_model.joblib; prediction inputs never leave the client.
"""
from __future__ import annotations

import json
import struct
import sys
import zlib
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, MODEL_PATH


def export_model() -> tuple[int, int]:
    pipeline = joblib.load(MODEL_PATH)
    preprocessor = pipeline.named_steps["preprocess"]
    forest = pipeline.named_steps["model"]
    encoder = preprocessor.named_transformers_["categorical"].named_steps["onehot"]
    categories = {
        feature: [str(value) for value in values]
        for feature, values in zip(CATEGORICAL_FEATURES, encoder.categories_)
    }

    offsets: list[int] = []
    packed = bytearray()
    for estimator in forest.estimators_:
        tree = estimator.tree_
        offsets.append(len(packed) // 28)
        for node in range(tree.node_count):
            packed.extend(
                struct.pack(
                    "<iiidd",
                    int(tree.children_left[node]),
                    int(tree.children_right[node]),
                    int(tree.feature[node]),
                    float(tree.threshold[node]),
                    float(tree.value[node, 0, 0]),
                )
            )
    offsets.append(len(packed) // 28)

    assets = Path(__file__).resolve().parent / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    binary_path = assets / "forest.bin.deflate"
    binary_path.write_bytes(zlib.compress(bytes(packed), level=9))
    manifest = {
        "format": "carcare-random-forest-v1",
        "tree_count": len(forest.estimators_),
        "node_count": offsets[-1],
        "tree_offsets": offsets,
        "node_bytes": 28,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "categorical_categories": categories,
        "feature_names": preprocessor.get_feature_names_out().tolist(),
        "target": "remaining_km_to_service",
        "source_model": "models/best_model.joblib",
        "compression": "zlib/deflate",
    }
    (assets / "model-manifest.json").write_text(
        json.dumps(manifest, separators=(",", ":")), encoding="utf-8"
    )
    return offsets[-1], len(packed)


if __name__ == "__main__":
    nodes, raw_size = export_model()
    print(f"Exported {nodes:,} tree nodes ({raw_size:,} bytes uncompressed).")
