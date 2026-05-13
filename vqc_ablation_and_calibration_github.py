# %% Cell 1: VQC Cross-Entropy Ablation + Calibration Metrics

import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "pennylane", "pennylane-lightning", "tqdm"])
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "pennylane-lightning[gpu]"], stderr=subprocess.DEVNULL)
except Exception:
    pass

import numpy as np
import pandas as pd
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.datasets import load_breast_cancer
from tqdm.auto import tqdm

CKP = "/kaggle/input/datasets/miitdaga/quantum-after-review-output/checkpoints"
PARK_PATH = "/kaggle/input/datasets/miitdaga/uci-parkinsons-disease-dataset/parkinsons.data"
DIAB_PATH = "/kaggle/input/datasets/organizations/uciml/pima-indians-diabetes-database/diabetes.csv"
VQC_EPOCHS = 30

print("=" * 60, flush=True)
print("PART 1: VQC Cross-Entropy vs MSE Ablation", flush=True)
print("=" * 60, flush=True)

print("\nLoading datasets...", flush=True)
df_p = pd.read_csv(PARK_PATH)
X_park = MinMaxScaler((0, np.pi)).fit_transform(
    PCA(8, random_state=42).fit_transform(
        StandardScaler().fit_transform(df_p.drop(["name", "status"], axis=1).values)))
y_park = df_p["status"].values

bc = load_breast_cancer()
X_bc = MinMaxScaler((0, np.pi)).fit_transform(
    PCA(8, random_state=42).fit_transform(
        StandardScaler().fit_transform(bc.data)))
y_bc = bc.target

df_d = pd.read_csv(DIAB_PATH)
X_diab = MinMaxScaler((0, np.pi)).fit_transform(
    PCA(8, random_state=42).fit_transform(
        StandardScaler().fit_transform(df_d.drop("Outcome", axis=1).values)))
y_diab = df_d["Outcome"].values

DATASETS = {
    "Parkinson's": {"X": X_park, "y": y_park},
    "Breast Cancer": {"X": X_bc, "y": y_bc},
    "Diabetes": {"X": X_diab, "y": y_diab},
}

n_qubits = 8
n_layers = 2

try:
    dev = qml.device("lightning.gpu", wires=n_qubits)
    print("Using lightning.gpu", flush=True)
except Exception:
    dev = qml.device("lightning.qubit", wires=n_qubits)
    print("Using lightning.qubit (CPU)", flush=True)

@qml.qnode(dev, interface="autograd", diff_method="adjoint")
def qnn_circuit(weights, x):
    qml.AngleEmbedding(x, wires=range(n_qubits))
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return qml.expval(qml.PauliZ(0))

def mse_cost(weights, X, Y_bipolar):
    preds = pnp.array([qnn_circuit(weights, x) for x in X])
    return pnp.mean((preds - Y_bipolar) ** 2)

def bce_cost(weights, X, Y_binary):
    raw = pnp.array([qnn_circuit(weights, x) for x in X])
    probs = (1 + raw) / 2
    probs = pnp.clip(probs, 1e-7, 1 - 1e-7)
    return -pnp.mean(Y_binary * pnp.log(probs) + (1 - Y_binary) * pnp.log(1 - probs))

def train_vqc(X_tr, y_tr, X_te, y_te, loss_type, epochs):
    pnp.random.seed(42)
    shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
    weights = pnp.random.random(size=shape, requires_grad=True)
    opt = qml.NesterovMomentumOptimizer(stepsize=0.1)

    X_tr_p = pnp.array(X_tr, requires_grad=False)
    X_te_p = pnp.array(X_te, requires_grad=False)

    if loss_type == "mse":
        Y_tr = pnp.array(y_tr * 2 - 1, requires_grad=False)
        cost_fn = lambda w: mse_cost(w, X_tr_p[bi], Y_tr[bi])
    else:
        Y_tr = pnp.array(y_tr.astype(float), requires_grad=False)
        cost_fn = lambda w: bce_cost(w, X_tr_p[bi], Y_tr[bi])

    train_accs, test_accs = [], []
    for ep in range(epochs):
        bi = np.random.RandomState(ep).randint(0, len(X_tr_p), 16)
        if loss_type == "mse":
            cost_fn_ep = lambda w: mse_cost(w, X_tr_p[bi], Y_tr[bi])
        else:
            cost_fn_ep = lambda w: bce_cost(w, X_tr_p[bi], Y_tr[bi])
        weights, _ = opt.step_and_cost(cost_fn_ep, weights)

        tr_preds = np.array([np.sign(float(qnn_circuit(weights, x))) for x in X_tr_p])
        te_preds = np.array([np.sign(float(qnn_circuit(weights, x))) for x in X_te_p])
        tr_labels = y_tr * 2 - 1
        te_labels = y_te * 2 - 1
        train_accs.append(accuracy_score(tr_labels, tr_preds))
        test_accs.append(accuracy_score(te_labels, te_preds))

    return train_accs, test_accs

print("\nTraining VQC with MSE and BCE on all datasets...\n", flush=True)

vqc_results = {}
for ds_name, ds in DATASETS.items():
    X, y = ds["X"], ds["y"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    for loss_type in ["mse", "bce"]:
        print(f"  {ds_name} — {loss_type.upper()} ({VQC_EPOCHS} epochs)...", flush=True)
        tr_accs, te_accs = train_vqc(X_tr, y_tr, X_te, y_te, loss_type, VQC_EPOCHS)
        vqc_results[(ds_name, loss_type)] = {"train": tr_accs, "test": te_accs}
        print(f"    Final test acc: {te_accs[-1]:.4f}", flush=True)

print("\n=== VQC Loss Function Comparison ===", flush=True)
print(f"{'Dataset':<20} {'MSE Test Acc':>15} {'BCE Test Acc':>15}", flush=True)
print("-" * 50, flush=True)
for ds_name in DATASETS:
    mse_acc = vqc_results[(ds_name, "mse")]["test"][-1]
    bce_acc = vqc_results[(ds_name, "bce")]["test"][-1]
    print(f"{ds_name:<20} {mse_acc:>15.4f} {bce_acc:>15.4f}", flush=True)

print("\n" + "=" * 60, flush=True)
print("PART 2: Calibration Metrics (ECE + Brier Score)", flush=True)
print("=" * 60, flush=True)

print("\nLoading cached Gram matrices...", flush=True)

GRAMS = {
    "Parkinson's": {
        "zz": np.load(f"{CKP}/gram_zz_parkinsons.npy"),
        "amp": np.load(f"{CKP}/gram_amp_parkinsons.npy"),
        "angle": np.load(f"{CKP}/gram_angle_parkinsons.npy"),
    },
    "Breast Cancer": {
        "zz": np.load(f"{CKP}/gram_zz_breast_cancer.npy"),
        "amp": np.load(f"{CKP}/gram_amp_breast_cancer.npy"),
        "angle": np.load(f"{CKP}/gram_angle_breast_cancer.npy"),
    },
    "Diabetes": {
        "zz": np.load(f"{CKP}/gram_zz_diabetes.npy"),
        "amp": np.load(f"{CKP}/gram_amp_diabetes.npy"),
        "angle": np.load(f"{CKP}/gram_angle_diabetes.npy"),
    },
}

def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        avg_conf = y_prob[mask].mean()
        avg_acc = y_true[mask].mean()
        ece += mask.sum() * abs(avg_conf - avg_acc)
    return ece / len(y_true)

def get_oof_predictions_calib(model_type, g_matrix, y_target, train_idx, n_splits=5):
    inner_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(train_idx))
    y_sub = y_target[train_idx]
    for itr_p, ival_p in inner_cv.split(np.zeros(len(train_idx)), y_sub):
        itr = train_idx[itr_p]
        ival = train_idx[ival_p]
        if model_type == "svm":
            m = SVC(kernel="precomputed")
            m.fit(g_matrix[itr][:, itr], y_target[itr])
            oof_preds[ival_p] = m.predict(g_matrix[ival][:, itr])
        elif model_type == "knn":
            m = KNeighborsClassifier(n_neighbors=5, metric="precomputed")
            m.fit(np.clip(1.0 - g_matrix[itr][:, itr], 0, None), y_target[itr])
            oof_preds[ival_p] = m.predict(np.clip(1.0 - g_matrix[ival][:, itr], 0, None))
    return oof_preds

print("Computing OOF probability predictions...\n", flush=True)

cv = StratifiedKFold(10, shuffle=True, random_state=42)

calib_results = []

for ds_name in DATASETS:
    print(f"  {ds_name}...", flush=True)
    X = DATASETS[ds_name]["X"]
    y = DATASETS[ds_name]["y"]
    gz = GRAMS[ds_name]["zz"]
    ga = GRAMS[ds_name]["angle"]
    gm = GRAMS[ds_name]["amp"]

    oof_prob_rbf = np.zeros(len(y))
    oof_prob_stack = np.zeros(len(y))
    oof_pred_rbf = np.zeros(len(y))
    oof_pred_stack = np.zeros(len(y))

    for tr, te in cv.split(X, y):
        clf_rbf = SVC(kernel="rbf", probability=True, random_state=42).fit(X[tr], y[tr])
        oof_prob_rbf[te] = clf_rbf.predict_proba(X[te])[:, 1]
        oof_pred_rbf[te] = clf_rbf.predict(X[te])

        p_sa = SVC(kernel="precomputed").fit(gm[tr][:,tr], y[tr]).predict(gm[te][:,tr])
        p_ka = KNeighborsClassifier(5, metric="precomputed").fit(
            np.clip(1-gm[tr][:,tr],0,None), y[tr]).predict(np.clip(1-gm[te][:,tr],0,None))
        p_sz = SVC(kernel="precomputed").fit(gz[tr][:,tr], y[tr]).predict(gz[te][:,tr])
        p_kn = KNeighborsClassifier(5, metric="precomputed").fit(
            np.clip(1-ga[tr][:,tr],0,None), y[tr]).predict(np.clip(1-ga[te][:,tr],0,None))

        oof_sa = get_oof_predictions_calib("svm", gm, y, tr)
        oof_ka = get_oof_predictions_calib("knn", gm, y, tr)
        oof_sz = get_oof_predictions_calib("svm", gz, y, tr)
        oof_kn = get_oof_predictions_calib("knn", ga, y, tr)
        X_meta_tr = np.column_stack((oof_sa, oof_ka, oof_sz, oof_kn))
        X_meta_te = np.column_stack((p_sa, p_ka, p_sz, p_kn))

        ml = LogisticRegression(random_state=42, class_weight="balanced", max_iter=1000)
        ml.fit(X_meta_tr, y[tr])
        oof_prob_stack[te] = ml.predict_proba(X_meta_te)[:, 1]
        oof_pred_stack[te] = ml.predict(X_meta_te)

    for model_name, probs, preds in [
        ("Classical RBF", oof_prob_rbf, oof_pred_rbf),
        ("OOF Stacking Ensemble", oof_prob_stack, oof_pred_stack),
    ]:
        ece = expected_calibration_error(y, probs)
        brier = brier_score_loss(y, probs)
        acc = accuracy_score(y, preds)
        calib_results.append({
            "Dataset": ds_name,
            "Model": model_name,
            "Accuracy": round(acc, 4),
            "ECE": round(ece, 4),
            "Brier Score": round(brier, 4),
        })

print("\n=== Calibration Metrics ===", flush=True)
df_calib = pd.DataFrame(calib_results)
for ds in DATASETS:
    print(f"\n{ds}:", flush=True)
    sub = df_calib[df_calib["Dataset"] == ds][["Model", "Accuracy", "ECE", "Brier Score"]]
    print(sub.to_string(index=False), flush=True)

print("\nDone.", flush=True)
