"""Generate failure-case Grad-CAM overlays for the PPT failure analysis slide.

Picks misclassifications prioritized by clinical importance:
  1. Melanoma predicted as benign (most dangerous miss)
  2. BCC / akiec predicted as benign (also serious)
  3. High-confidence wrong predictions on any class

For each case, exports:
  - The original image
  - The Grad-CAM overlay
  - A side-by-side comparison PNG
  - A metadata JSON for filling in explanations

Outputs land in models/failure_cases/.
"""

import argparse
import json
import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

from utils.preprocessing import (
    DISEASE_NAMES,
    IMAGE_SIZE,
    LABEL_NAMES,
    build_tf_dataset,
    find_last_conv_layer,
    load_metadata,
    load_trained_model,
    make_gradcam_heatmap,
    prepare_inference_image,
    resolve_image_paths,
    split_metadata_grouped,
    split_metadata_image_stratified,
)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "dataset")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Clinical severity rank — lower is more dangerous to miss
CLASS_SEVERITY = {
    "mel": 0,    # melanoma — deadly
    "bcc": 1,    # basal cell carcinoma — common skin cancer
    "akiec": 2,  # actinic keratoses — pre-cancerous
    "bkl": 3,
    "df": 3,
    "vasc": 3,
    "nv": 4,     # benign nevus
}

BENIGN_CLASSES = {"nv", "bkl", "df", "vasc"}


def overlay_gradcam(image_np: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Return an RGB float [0,1] image with the Grad-CAM heatmap overlaid."""
    img = image_np.astype(np.float32)
    if img.max() > 1.5:
        img = img / 255.0
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    merged = np.clip((1 - alpha) * img + alpha * heatmap_color, 0, 1)
    return merged


def case_priority(true_label: str, pred_label: str, confidence: float) -> tuple:
    """Lower tuple = higher priority for selection."""
    severity_miss = CLASS_SEVERITY.get(true_label, 5)
    benign_misprediction = 0 if (true_label not in BENIGN_CLASSES and pred_label in BENIGN_CLASSES) else 1
    return (severity_miss, benign_misprediction, -confidence)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=os.path.join(MODELS_DIR, "cnn_model.keras"))
    parser.add_argument(
        "--split-strategy",
        choices=["grouped", "image-stratified"],
        default="grouped",
        help="Which test set to draw failure cases from",
    )
    parser.add_argument("--num-cases", type=int, default=4)
    parser.add_argument("--out-dir", default=os.path.join(MODELS_DIR, "failure_cases"))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading model: {args.model_path}")
    model = load_trained_model(args.model_path)
    last_conv = find_last_conv_layer(model)

    image_dirs = [
        os.path.join(DATA_DIR, "HAM10000_images_part_1"),
        os.path.join(DATA_DIR, "HAM10000_images_part_2"),
    ]
    metadata_path = os.path.join(DATA_DIR, "HAM10000_metadata.csv")

    print(f"Loading metadata and {args.split_strategy} test split")
    meta = load_metadata(metadata_path)
    meta = resolve_image_paths(meta, image_dirs)
    if args.split_strategy == "image-stratified":
        _, _, test_df = split_metadata_image_stratified(meta, seed=42)
    else:
        _, _, test_df = split_metadata_grouped(meta, seed=42)

    test_ds = build_tf_dataset(test_df, image_size=IMAGE_SIZE, batch_size=32)

    print("Running inference on the test set...")
    probs = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(probs, axis=1)
    y_conf = probs.max(axis=1)
    y_true = test_df["label"].to_numpy(dtype=np.int32)
    image_paths = test_df["image_path"].tolist()

    mismatches = np.where(y_pred != y_true)[0]
    print(f"Found {len(mismatches)} misclassifications out of {len(y_true)} test samples")

    ranked = sorted(
        mismatches,
        key=lambda i: case_priority(
            true_label=LABEL_NAMES[int(y_true[i])],
            pred_label=LABEL_NAMES[int(y_pred[i])],
            confidence=float(y_conf[i]),
        ),
    )

    selected = ranked[: args.num_cases]
    print(f"\nSelected {len(selected)} failure cases (ranked by clinical importance):")

    cases_meta = []
    for rank, idx in enumerate(selected, start=1):
        true_lbl = LABEL_NAMES[int(y_true[idx])]
        pred_lbl = LABEL_NAMES[int(y_pred[idx])]
        conf = float(y_conf[idx])
        true_dx = DISEASE_NAMES[true_lbl]
        pred_dx = DISEASE_NAMES[pred_lbl]
        img_path = image_paths[idx]

        print(f"  Case {rank}: {true_lbl} ({true_dx}) → predicted {pred_lbl} ({pred_dx}) @ {conf:.1%} conf")
        print(f"           image: {os.path.basename(img_path)}")

        pil_img = Image.open(img_path).convert("RGB")
        input_arr = prepare_inference_image(pil_img, image_size=IMAGE_SIZE)
        heatmap = make_gradcam_heatmap(input_arr, model, last_conv)
        overlay = overlay_gradcam(input_arr[0], heatmap)

        original_display = np.array(pil_img.resize(IMAGE_SIZE)).astype(np.float32) / 255.0

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(original_display)
        axes[0].set_title(f"Original — true: {true_dx}", fontsize=11)
        axes[0].axis("off")

        axes[1].imshow(overlay)
        axes[1].set_title(f"Grad-CAM — predicted: {pred_dx} ({conf:.1%})", fontsize=11)
        axes[1].axis("off")

        fig.suptitle(f"Case {rank}: {true_lbl} mispredicted as {pred_lbl}", fontsize=13, fontweight="bold")
        plt.tight_layout()

        combined_path = os.path.join(args.out_dir, f"case_{rank}_{true_lbl}_as_{pred_lbl}.png")
        plt.savefig(combined_path, dpi=160, bbox_inches="tight")
        plt.close(fig)

        overlay_path = os.path.join(args.out_dir, f"case_{rank}_{true_lbl}_as_{pred_lbl}_gradcam_only.png")
        plt.imsave(overlay_path, overlay)

        original_path = os.path.join(args.out_dir, f"case_{rank}_{true_lbl}_as_{pred_lbl}_original.png")
        plt.imsave(original_path, original_display)

        cases_meta.append({
            "rank": rank,
            "true_label": true_lbl,
            "predicted_label": pred_lbl,
            "true_disease": true_dx,
            "predicted_disease": pred_dx,
            "confidence": round(conf, 4),
            "image_filename": os.path.basename(img_path),
            "image_id": test_df.iloc[idx]["image_id"],
            "files": {
                "combined": os.path.basename(combined_path),
                "gradcam_only": os.path.basename(overlay_path),
                "original": os.path.basename(original_path),
            },
            "explanation": "TODO — fill in why the model failed (hair, contrast, ambiguous boundary, etc.)",
        })

    meta_path = os.path.join(args.out_dir, "failure_cases_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(cases_meta, f, indent=2)

    print(f"\n[done] Output directory: {args.out_dir}")
    print(f"       Combined images: case_N_<true>_as_<pred>.png  (drop these straight into the PPT)")
    print(f"       Metadata:        {os.path.basename(meta_path)}  (fill in the 'explanation' field for each case)")


if __name__ == "__main__":
    main()
