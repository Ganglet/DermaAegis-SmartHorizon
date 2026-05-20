import argparse
import json
import os
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from utils.preprocessing import (
    DISEASE_NAMES,
    IMAGE_SIZE,
    LABEL_NAMES,
    build_tf_dataset,
    load_metadata,
    load_trained_model,
    resolve_image_paths,
    split_metadata_grouped,
    split_metadata_image_stratified,
)


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "dataset")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

parser = argparse.ArgumentParser(description="Evaluate a HAM10000 checkpoint on the test split with TTA")
parser.add_argument("--model-path", default=None, help="Path to a .keras checkpoint; defaults to latest .best.keras")
parser.add_argument(
    "--split-strategy",
    choices=["grouped", "image-stratified"],
    default="image-stratified",
    help="Dataset split strategy used for evaluation — MUST match what the model was trained on",
)
parser.add_argument("--tta-runs", type=int, default=4)
parser.add_argument("--tag", type=str, default="", help="Suffix for output artifacts")
args = parser.parse_args()

if args.model_path:
    MODEL_PATH = os.path.abspath(args.model_path)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {MODEL_PATH}")
else:
    best_candidates = [
        os.path.join(MODELS_DIR, name)
        for name in os.listdir(MODELS_DIR)
        if name.endswith(".best.keras")
    ]
    if not best_candidates:
        raise FileNotFoundError("No .best.keras checkpoint found in models directory")
    MODEL_PATH = max(best_candidates, key=os.path.getmtime)

tag = args.tag or args.split_strategy
print(f"Evaluating checkpoint: {MODEL_PATH}")
print(f"Split strategy: {args.split_strategy}")
print(f"TTA runs: {args.tta_runs}")
print(f"Artifact tag: {tag}")

image_dirs = [
    os.path.join(DATA_DIR, "HAM10000_images_part_1"),
    os.path.join(DATA_DIR, "HAM10000_images_part_2"),
]
metadata_path = os.path.join(DATA_DIR, "HAM10000_metadata.csv")

meta = load_metadata(metadata_path)
meta = resolve_image_paths(meta, image_dirs)
if args.split_strategy == "image-stratified":
    _, _, test_df = split_metadata_image_stratified(meta, seed=42)
else:
    _, _, test_df = split_metadata_grouped(meta, seed=42)

y_test = test_df["label"].to_numpy(dtype=np.int32)
test_ds = build_tf_dataset(test_df, image_size=IMAGE_SIZE, batch_size=32)

model = load_trained_model(MODEL_PATH)

# Plain accuracy (no TTA)
print("\nRunning plain inference (no TTA)...")
plain_probs = model.predict(test_ds, verbose=0)
y_pred_plain = np.argmax(plain_probs, axis=1)
plain_acc = float((y_pred_plain == y_test).mean())
print(f"Plain test accuracy: {plain_acc:.4f}")

# TTA: average predictions over flipped variants
print(f"\nRunning TTA with {args.tta_runs} passes...")
tta_sum = None
for run_idx in range(args.tta_runs):
    run_probs = []
    for batch_images, _ in test_ds:
        x = batch_images
        if run_idx == 1:
            x = tf.image.flip_left_right(x)
        elif run_idx == 2:
            x = tf.image.flip_up_down(x)
        elif run_idx == 3:
            x = tf.image.flip_left_right(tf.image.flip_up_down(x))
        run_probs.append(model(x, training=False).numpy())
    run_probs = np.concatenate(run_probs, axis=0)
    tta_sum = run_probs if tta_sum is None else tta_sum + run_probs

probs = tta_sum / args.tta_runs
y_pred = np.argmax(probs, axis=1)
tta_acc = float((y_pred == y_test).mean())
print(f"TTA test accuracy ({args.tta_runs}x): {tta_acc:.4f}")

target_names = [LABEL_NAMES[i] for i in range(len(LABEL_NAMES))]
print("\nClassification report (with TTA):")
print(classification_report(y_test, y_pred, target_names=target_names, digits=4))

report = classification_report(
    y_test,
    y_pred,
    target_names=target_names,
    digits=4,
    output_dict=True,
)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
fig = plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", xticklabels=target_names, yticklabels=target_names)
plt.title(f"HAM10000 Confusion Matrix — {tag} (TTA {args.tta_runs}x, acc {tta_acc:.3f})")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
cm_path = os.path.join(MODELS_DIR, f"confusion_matrix_{tag}.png")
plt.savefig(cm_path, dpi=160)
plt.close(fig)
print(f"\nConfusion matrix saved: {cm_path}")

# Save report + metadata for this evaluation
report_path = os.path.join(MODELS_DIR, f"classification_report_{tag}.json")
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

eval_meta = {
    "evaluated_checkpoint": MODEL_PATH,
    "split_strategy": args.split_strategy,
    "tta_runs": args.tta_runs,
    "plain_test_accuracy": plain_acc,
    "tta_test_accuracy": tta_acc,
    "classification_report": report,
    "label_names": LABEL_NAMES,
    "disease_names": DISEASE_NAMES,
    "evaluated_at": datetime.utcnow().isoformat() + "Z",
}
eval_meta_path = os.path.join(MODELS_DIR, f"eval_metadata_{tag}.json")
with open(eval_meta_path, "w", encoding="utf-8") as f:
    json.dump(eval_meta, f, indent=2)
print(f"Eval metadata saved: {eval_meta_path}")
print(f"Report saved: {report_path}")
