# Adaptive Quantum Kernel Selection via Leakage-Free Stacking for Clinical Diagnostics on NISQ Hardware

Experimental pipeline for the paper submitted to **Scientific Reports**.

## What this does

Routes clinical patient data through three quantum feature maps (Angle, Amplitude, ZZ-entanglement), trains QSVM and QKNN classifiers on the resulting Gram matrices, and feeds their predictions into a Logistic Regression meta-learner under strict nested cross-validation to prevent data leakage. Evaluated on three datasets: Parkinson's (195 patients), Breast Cancer (569), and Diabetes (768).

## Results at a glance

| Dataset | Classical RBF Specificity | Quantum Ensemble Specificity | p-value |
|---|---|---|---|
| Parkinson's | 0.585 | **0.813** | <0.001 |
| Breast Cancer | 0.948 | 0.943 | >0.05 |
| Diabetes (Recall) | 0.553 | **0.621** | 0.004 |


## Requirements

- Python 3.10+
- Kaggle environment with GPU (T4 or better) recommended


Install dependencies:
```bash
pip install pennylane pennylane-lightning shap lime statsmodels xgboost umap-learn tqdm
```

See `requirements.txt` for pinned versions matching the paper.

## Datasets

All three datasets are publicly available:

1. **UCI Parkinson's**: [UCI Repository](https://archive.ics.uci.edu/dataset/174/parkinsons)
2. **Wisconsin Breast Cancer**: Available via `sklearn.datasets.load_breast_cancer()` and [UCI Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
3. **Pima Indians Diabetes**: [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

Update the paths on lines 225 and 252 of `quantum_clinical_github.py` to match your local dataset locations:
```python
park_path = "/your/path/to/parkinsons.data"
diab_path = "/your/path/to/diabetes.csv"
```

## Running

The script is organized as cells (marked with `# %% cell N:`) for use in Jupyter/Kaggle notebooks. Copy-paste each cell into a notebook, or run the full script:

```bash
python quantum_clinical_github.py
```

### Pilot mode

Set `PILOT_MODE = True` at the top of the file for a quick smoke test. This reduces VQC epochs from 30 to 5, bootstrap iterations from 1000 to 50, and noise subsamples from 60 to 30. All non-VQC results are identical between pilot and full runs.

### Outputs

Results are saved to three directories under the working directory:

```
figures/          # All plots at 600 DPI (PNG)
tables/           # All numerical results (CSV)
checkpoints/      # Gram matrices (NPY) for caching
```

Gram matrices are cached as `.npy` files. On subsequent runs, they load from cache instead of recomputing.

## Additional analyses

`base_model_selection_mcnemar.py` reproduces the McNemar pairwise redundancy tests
used to select the four base models from the six-model candidate space (Section 4.3).
Requires precomputed Gram matrices in `checkpoints/`.

`vqc_ablation_and_calibration_github.py` runs two supplementary analyses:
- **VQC loss function ablation:** Trains the VQC with both MSE and cross-entropy loss,
  confirming that the barren plateau is the cause of failure, not the loss choice (Section 4.5).
- **Calibration metrics:** Computes ECE and Brier scores for the Classical RBF and OOF
  Stacking Ensemble using cached Gram matrices (Section 5).

`platt_scaling_github.py` applies post-hoc Platt scaling to the OOF Stacking Ensemble
and Classical RBF, reporting ECE and Brier scores before and after calibration (Section 5).
Requires precomputed Gram matrices in `checkpoints/`.

## Pipeline structure

| Cell | What |
|---|---|
| 1-2 | Configuration, imports, GPU detection |
| 3-4 | Quantum circuit definitions, helper functions |
| 5 | Data preparation (all 3 datasets) |
| 6 | Scree plot (PCA component justification) |
| 7 | Kernel PCA / UMAP comparison |
| 8 | Gram matrix computation (9 matrices, cached) |
| 9 | VQC standalone learning curves |
| 10 | 2D decision boundaries |
| 11 | Master evaluation (10 models, parametric + bootstrap) |
| 12 | Classical ensemble baselines (XGBoost, RF, Stacking) |
| 13 | SHAP analysis |
| 14 | LIME + Permutation Importance |
| 15 | Circuit timing, ROC/PR, McNemar's, Ablation |
| 16 | Noise robustness + Clinical feature backtracking |
| 17 | Master bar chart |


## License

This code is provided for academic research purposes. See the paper for full methodological details.
