# Group 5: Fruit Freshness Classification, Comparison of CNN Model Performance Using Image Preprocessing Techniques on EfficientNetB0 and ResNet50V2

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
| **ResNet50V2 — WITH preprocessing** ★ | **0.9444** | **0.9500** | **0.9444** | **0.9443** |
| ResNet50V2 — WITHOUT preprocessing | 0.9167 | 0.9286 | 0.9167 | 0.9161 |

**ResNet50V2 with preprocessing** came out on top: the only variant with zero false negatives on the fresh class (confusion matrix: 18/18 fresh correctly classified, only 2/18 rotten misclassified as fresh). Preprocessing consistently improved both architectures (+2.8% F1-score on average), and ResNet50V2's skip connections handled subtle visual cues (gradual discoloration, small spots) better than EfficientNetB0's compound scaling. The most common error across all models was misclassifying partially-rotten fruit as fresh, likely because CLAHE-enhanced contrast still can't fully compensate for cases where most of the fruit's surface still looks fresh.

## Tech Stack
- **Deep Learning:** TensorFlow, Keras
- **Image Processing:** OpenCV (cv2), Pillow (PIL)
- **Evaluation:** Scikit-Learn (classification_report, confusion_matrix, accuracy/precision/recall/F1)
- **Data Handling:** NumPy, h5py, Pathlib
- **Visualization:** Matplotlib, Seaborn
- **Environment:** Python, Google Colab (GPU)

## File Structure
```
fruit-freshness-classification/
├── CompVis_Fruit_Freshness.ipynb   ← Main notebook: preprocessing, training, evaluation, TTA
├── Dataset.zip                     ← Fruits Quality: Fresh VS Rotten (Kaggle), extract to get train/valid/test
├── Outputs/                        ← Generated results
│   ├── confusion_matrices.png      ← Confusion matrix for all 4 model variants
│   ├── metric_comparison.png       ← Accuracy/Precision/Recall/F1 bar chart comparison
│   ├── training_curves.png         ← Accuracy & loss curves (Head Training + Fine-Tuning)
│   ├── preprocessing_diff.png      ← Visual comparison: WITH vs WITHOUT preprocessing
│   ├── preds_EfficientNetB0_With_Preprocessing.png  ← Sample predictions, EfficientNetB0
│   └── preds_ResNet50V2_With_Preprocessing.png      ← Sample predictions, ResNet50V2
└── README.md
```

---
Dataset: Abdoun, N. (2023). *Fruits quality: Fresh vs rotten* [Dataset]. Kaggle.
