# Group 5 — Fruit Freshness Classification: Comparison of CNN Model Performance Using Image Preprocessing Techniques on EfficientNetB0 and ResNet50V2

A computer vision project comparing CNN model performance for binary fruit freshness classification (**fresh** vs **rotten**), using transfer learning on **EfficientNetB0** and **ResNet50V2**, with and without image preprocessing (Gaussian Blur + CLAHE).

## Background

Manual fruit quality inspection is slow, inconsistent, and easily affected by lighting conditions. This project explores whether a CNN-based system, combined with image enhancement techniques, can classify fruit freshness more reliably than a baseline model without preprocessing.

## Dataset

| Dataset | Images | Classes | Purpose |
|---|---|---|---|
| [Fruits Quality: Fresh vs Rotten](https://www.kaggle.com/datasets/nourabdoun/fruits-quality-fresh-vs-rotten) (Kaggle, Nour Abdoun) | 359 | Fresh (181) · Rotten (178) | Model training & evaluation |

- **Split:** Stratified 80:10:10. Train 287 (145 fresh, 142 rotten), Validation 36 (18/18), Test 36 (18/18)
- **Fruit variety:** strawberry, apple, banana, kiwi, mango, grape, cherry, pear, peach, orange, etc.
- Class weight computed automatically (`balanced`) to keep training balanced.

> **Note:** The dataset is included in this repository as `Dataset.zip`. Extract it to get the `train/`, `valid/`, and `test/` split.

## Approach

Two CNN architectures are compared head-to-head, each under two scenarios:

- **WITH Preprocessing:** Gaussian Blur (3×3 kernel, σ=0.5), then CLAHE (clipLimit=2.5, tileGrid=8×8 on the L channel/LAB), then architecture-specific normalization
- **WITHOUT Preprocessing:** resize + standard normalization (baseline)

This produces **4 model variants**: EfficientNetB0 WITH/WITHOUT and ResNet50V2 WITH/WITHOUT.

### Model Architecture

| | EfficientNetB0 | ResNet50V2 |
|---|---|---|
| Approach | Compound scaling (depth × width × resolution) | Residual/skip connections |
| Parameters | ~5.3M | ~25.6M |
| Normalization | pass-through [0, 255] | rescale to [-1, 1] |
| Classifier Head | GAP, BN, Dropout(0.5), Dense(2, softmax) | GAP, BN, Dense(512, ReLU), Drop(0.5), Dense(256, ReLU), Drop(0.3), Dense(2, softmax) |

### Training Strategy
Training runs in two phases:
- **Phase 1, Head Training:** backbone frozen, max 30 epochs, LR 1e-3, strong augmentation (rotation 40°, zoom 0.3, brightness 0.6–1.4, channel shift, shear, flip)
- **Phase 2, Fine-Tuning:** unfreeze 20 layers (EfficientNet) / 50 layers (ResNet), max 25 epochs, LR 5e-5, light augmentation (rotation 15°, horizontal flip, zoom 0.1, brightness 0.85–1.15)

**Optimizer:** Adam. **Loss:** CategoricalCrossentropy (label smoothing 0.1). **Callbacks:** EarlyStopping (patience=10), ReduceLROnPlateau (factor=0.4, patience=5)

Final evaluation uses **Test Time Augmentation (TTA)** with 7 steps (horizontal flip, rotation ±12°, zoom ±0.08, brightness 0.9–1.1). Predictions are averaged across all 7 for more stable results.

## Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| EfficientNetB0 — WITH preprocessing | 0.8889 | 0.8889 | 0.8889 | 0.8889 |
| EfficientNetB0 — WITHOUT preprocessing | 0.8611 | 0.8622 | 0.8611 | 0.8610 |
| **ResNet50V2 — WITH preprocessing**  | **0.9444** | **0.9500** | **0.9444** | **0.9443** |
| ResNet50V2 — WITHOUT preprocessing | 0.9167 | 0.9286 | 0.9167 | 0.9161 |

**ResNet50V2 with preprocessing** came out on top: the only variant with zero false negatives on the fresh class (confusion matrix: 18/18 fresh correctly classified, only 2/18 rotten misclassified as fresh). Preprocessing consistently improved both architectures (+2.8% F1-score on average), and ResNet50V2's skip connections handled subtle visual cues (gradual discoloration, small spots) better than EfficientNetB0's compound scaling. The most common error across all models was misclassifying partially-rotten fruit as fresh, likely because CLAHE-enhanced contrast still can't fully compensate for cases where most of the fruit's surface still looks fresh.

## MLflow Experiment Tracking

The MLflow pipeline for this project is included as a separate deliverable.

| File | Description |
|---|---|
| `MLflow.ipynb` | Notebook that logs all 4 model variants as MLflow runs — params, metrics, and model artifacts |
| `artifacts.zip` | Pre-run MLflow tracking data: `mlflow.db` (SQLite) + `mlruns/` with logged model artifacts |

**Experiment name:** `Fruit Freshness Classification - EfficientNetB0 vs ResNet50V2`

**Runs logged (4 total):**

| Run name | test_f1 | test_accuracy |
|---|---|---|
| `EfficientNetB0_with_preprocessing` | 0.8889 | 0.8889 |
| `EfficientNetB0_without_preprocessing` | 0.8610 | 0.8611 |
| `ResNet50V2_with_preprocessing` | 0.9443 | 0.9444 |
| `ResNet50V2_without_preprocessing` | 0.9161 | 0.9167 |

Each run logs: architecture, preprocessing config, all training hyperparameters (phases 1 & 2), TTA steps, dataset split sizes, test metrics (accuracy/precision/recall/F1), confusion matrix values (TP/TN/FP/FN), final validation accuracy/loss, and a stub model artifact with the correct classifier head.

**To view results locally:**
```bash
# Extract artifacts.zip first, then:
pip install mlflow
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://127.0.0.1:5000
```

**To re-run the pipeline from scratch:**
```bash
jupyter notebook MLflow.ipynb
# Requires: tensorflow, mlflow, numpy
```

> **Note:** `MLflow.ipynb` uses stub models (same head architecture, randomly initialized weights) to keep the demo fast and reproducible. The actual trained weights are in `keras_models/`. Metrics logged are the real results from Table 1 of the paper.

## Tech Stack
- **Deep Learning:** TensorFlow, Keras
- **Image Processing:** OpenCV (cv2), Pillow (PIL)
- **Evaluation:** Scikit-Learn (classification_report, confusion_matrix, accuracy/precision/recall/F1)
- **Data Handling:** NumPy, h5py, Pathlib
- **Visualization:** Matplotlib, Seaborn
- **Experiment Tracking:** MLflow
- **Environment:** Python, Google Colab (GPU)

## File Structure
```
fruit-freshness-classification/
├── CompVis_Fruit_Freshness.ipynb   ← Main notebook: preprocessing, training, evaluation, TTA
├── MLflow.ipynb                    ← MLflow tracking pipeline (logs all 4 variants)
├── artifacts.zip                   ← MLflow artifacts: mlflow.db + mlruns/ (extract before use)
├── Dataset.zip                     ← Fruits Quality: Fresh VS Rotten (Kaggle), extract to get train/valid/test
├── fruit_splits.h5                 ← Stratified train/valid/test split indices
├── keras_models/                   ← Trained model weights (stored via Git LFS)
│   ├── eff_with_p1.keras           ← EfficientNetB0 WITH preprocessing, Phase 1
│   ├── eff_with_p2.keras           ← EfficientNetB0 WITH preprocessing, Phase 2 (fine-tuned)
│   ├── eff_wo_p1.keras             ← EfficientNetB0 WITHOUT preprocessing, Phase 1
│   ├── eff_wo_p2.keras             ← EfficientNetB0 WITHOUT preprocessing, Phase 2 (fine-tuned)
│   ├── res_with_p1.keras           ← ResNet50V2 WITH preprocessing, Phase 1
│   ├── res_with_p2.keras           ← ResNet50V2 WITH preprocessing, Phase 2 (fine-tuned)
│   ├── res_wo_p1.keras             ← ResNet50V2 WITHOUT preprocessing, Phase 1
│   └── res_wo_p2.keras             ← ResNet50V2 WITHOUT preprocessing, Phase 2 (fine-tuned)
├── Outputs/                        ← Generated results
│   ├── confusion_matrices.png      ← Confusion matrix for all 4 model variants
│   ├── metric_comparison.png       ← Accuracy/Precision/Recall/F1 bar chart comparison
│   ├── training_curves.png         ← Accuracy & loss curves (Head Training + Fine-Tuning)
│   ├── preprocessing_diff.png      ← Visual comparison: WITH vs WITHOUT preprocessing
│   ├── preds_EfficientNetB0_With_Preprocessing.png  ← Sample predictions, EfficientNetB0
│   └── preds_ResNet50V2_With_Preprocessing.png      ← Sample predictions, ResNet50V2
└── README.md
```

> **⚠️ Note:** The `.keras` model files in `keras_models/` are stored using **Git LFS**. To download the actual model files, please **clone** this repository instead of downloading the ZIP:
> ```bash
> git clone https://github.com/vebynovalisa/fruit-freshness-classification.git
> ```
> Downloading via the GitHub ZIP button will only give you pointer files, not the actual models.

---
Dataset: Abdoun, N. (2023). *Fruits quality: Fresh vs rotten* [Dataset]. Kaggle.
