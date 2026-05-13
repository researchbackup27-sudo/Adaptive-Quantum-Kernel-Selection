import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_breast_cancer

CKP = "/kaggle/input/datasets/miitdaga/quantum-after-review-output/checkpoints"
PARK_PATH = "/kaggle/input/datasets/miitdaga/uci-parkinsons-disease-dataset/parkinsons.data"
DIAB_PATH = "/kaggle/input/datasets/organizations/uciml/pima-indians-diabetes-database/diabetes.csv"

def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        ece += mask.sum() * abs(y_prob[mask].mean() - y_true[mask].mean())
    return ece / len(y_true)

def get_oof_preds(model_type, g_matrix, y, train_idx, n_splits=5):
    inner_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(train_idx))
    y_sub = y[train_idx]
    for itr_p, ival_p in inner_cv.split(np.zeros(len(train_idx)), y_sub):
        itr, ival = train_idx[itr_p], train_idx[ival_p]
        if model_type == "svm":
            m = SVC(kernel="precomputed").fit(g_matrix[itr][:,itr], y[itr])
            oof[ival_p] = m.predict(g_matrix[ival][:,itr])
        else:
            m = KNeighborsClassifier(5, metric="precomputed").fit(
                np.clip(1-g_matrix[itr][:,itr],0,None), y[itr])
            oof[ival_p] = m.predict(np.clip(1-g_matrix[ival][:,itr],0,None))
    return oof

print("Loading data...", flush=True)
y_park = pd.read_csv(PARK_PATH)["status"].values
y_bc = load_breast_cancer().target
y_diab = pd.read_csv(DIAB_PATH)["Outcome"].values

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

X_park = MinMaxScaler((0,np.pi)).fit_transform(PCA(8,random_state=42).fit_transform(
    StandardScaler().fit_transform(pd.read_csv(PARK_PATH).drop(["name","status"],axis=1).values)))
X_bc = MinMaxScaler((0,np.pi)).fit_transform(PCA(8,random_state=42).fit_transform(
    StandardScaler().fit_transform(load_breast_cancer().data)))
X_diab = MinMaxScaler((0,np.pi)).fit_transform(PCA(8,random_state=42).fit_transform(
    StandardScaler().fit_transform(pd.read_csv(DIAB_PATH).drop("Outcome",axis=1).values)))

DATASETS = {
    "Parkinson's": {"X": X_park, "y": y_park,
        "zz": np.load(f"{CKP}/gram_zz_parkinsons.npy"),
        "amp": np.load(f"{CKP}/gram_amp_parkinsons.npy"),
        "angle": np.load(f"{CKP}/gram_angle_parkinsons.npy")},
    "Breast Cancer": {"X": X_bc, "y": y_bc,
        "zz": np.load(f"{CKP}/gram_zz_breast_cancer.npy"),
        "amp": np.load(f"{CKP}/gram_amp_breast_cancer.npy"),
        "angle": np.load(f"{CKP}/gram_angle_breast_cancer.npy")},
    "Diabetes": {"X": X_diab, "y": y_diab,
        "zz": np.load(f"{CKP}/gram_zz_diabetes.npy"),
        "amp": np.load(f"{CKP}/gram_amp_diabetes.npy"),
        "angle": np.load(f"{CKP}/gram_angle_diabetes.npy")},
}

print("Computing calibration with Platt scaling...\n", flush=True)

cv_outer = StratifiedKFold(10, shuffle=True, random_state=42)
cv_calib = StratifiedKFold(5, shuffle=True, random_state=42)

results = []

for ds_name, ds in DATASETS.items():
    print(f"  {ds_name}...", flush=True)
    X, y = ds["X"], ds["y"]
    gz, ga, gm = ds["zz"], ds["angle"], ds["amp"]

    oof_prob_rbf_raw = np.zeros(len(y))
    oof_prob_rbf_platt = np.zeros(len(y))
    oof_prob_stack_raw = np.zeros(len(y))
    oof_prob_stack_platt = np.zeros(len(y))

    for tr, te in cv_outer.split(X, y):
        clf_rbf = SVC(kernel="rbf", probability=True, random_state=42).fit(X[tr], y[tr])
        oof_prob_rbf_raw[te] = clf_rbf.predict_proba(X[te])[:,1]

        clf_rbf_cal = CalibratedClassifierCV(clf_rbf, cv=5, method="sigmoid").fit(X[tr], y[tr])
        oof_prob_rbf_platt[te] = clf_rbf_cal.predict_proba(X[te])[:,1]

        p_sa = SVC(kernel="precomputed").fit(gm[tr][:,tr], y[tr]).predict(gm[te][:,tr])
        p_ka = KNeighborsClassifier(5, metric="precomputed").fit(
            np.clip(1-gm[tr][:,tr],0,None), y[tr]).predict(np.clip(1-gm[te][:,tr],0,None))
        p_sz = SVC(kernel="precomputed").fit(gz[tr][:,tr], y[tr]).predict(gz[te][:,tr])
        p_kn = KNeighborsClassifier(5, metric="precomputed").fit(
            np.clip(1-ga[tr][:,tr],0,None), y[tr]).predict(np.clip(1-ga[te][:,tr],0,None))

        oof_sa = get_oof_preds("svm", gm, y, tr)
        oof_ka = get_oof_preds("knn", gm, y, tr)
        oof_sz = get_oof_preds("svm", gz, y, tr)
        oof_kn = get_oof_preds("knn", ga, y, tr)
        X_meta_tr = np.column_stack((oof_sa, oof_ka, oof_sz, oof_kn))
        X_meta_te = np.column_stack((p_sa, p_ka, p_sz, p_kn))

        ml_raw = LogisticRegression(random_state=42, class_weight="balanced", max_iter=1000)
        ml_raw.fit(X_meta_tr, y[tr])
        oof_prob_stack_raw[te] = ml_raw.predict_proba(X_meta_te)[:,1]

        ml_cal = CalibratedClassifierCV(ml_raw, cv=5, method="sigmoid").fit(X_meta_tr, y[tr])
        oof_prob_stack_platt[te] = ml_cal.predict_proba(X_meta_te)[:,1]

    for model, raw, cal in [
        ("Classical RBF", oof_prob_rbf_raw, oof_prob_rbf_platt),
        ("OOF Stacking Ensemble", oof_prob_stack_raw, oof_prob_stack_platt)]:
        results.append({
            "Dataset": ds_name, "Model": model,
            "ECE (raw)": round(expected_calibration_error(y, raw), 4),
            "ECE (Platt)": round(expected_calibration_error(y, cal), 4),
            "Brier (raw)": round(brier_score_loss(y, raw), 4),
            "Brier (Platt)": round(brier_score_loss(y, cal), 4),
        })

print("\n=== Calibration: Before vs After Platt Scaling ===\n", flush=True)
df = pd.DataFrame(results)
for ds in DATASETS:
    print(f"{ds}:", flush=True)
    sub = df[df["Dataset"]==ds][["Model","ECE (raw)","ECE (Platt)","Brier (raw)","Brier (Platt)"]]
    print(sub.to_string(index=False), flush=True)
    print(flush=True)

print("Done.", flush=True)
