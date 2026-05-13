# %% cell 1: Configuration and Installation
PILOT_MODE = False

VQC_EPOCHS_STANDALONE = 5 if PILOT_MODE else 30
VQC_EPOCHS_INFOLD     = 5 if PILOT_MODE else 15
N_BOOTSTRAPS          = 50 if PILOT_MODE else 1000
NOISE_SUBSAMPLE       = 30 if PILOT_MODE else 60
NOISE_LEVELS          = [0.00, 0.10] if PILOT_MODE else [0.00, 0.05, 0.10]
LIME_SAMPLES          = 10 if PILOT_MODE else 50
SUFFIX                = "_pilot" if PILOT_MODE else ""

import subprocess, sys
pkgs = [
    "pennylane", "pennylane-lightning", "shap", "lime",
    "statsmodels", "xgboost", "umap-learn", "tqdm"
]
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + pkgs)

try:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "pennylane-lightning[gpu]"],
        stderr=subprocess.DEVNULL
    )
    _gpu_install = True
except Exception:
    _gpu_install = False

print(f"PILOT_MODE = {PILOT_MODE}")
print(f"VQC epochs (standalone/infold): {VQC_EPOCHS_STANDALONE}/{VQC_EPOCHS_INFOLD}")
print(f"Bootstrap iterations: {N_BOOTSTRAPS}")

# %% cell 2: Imports and GPU detection
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time, os, warnings, pickle
warnings.filterwarnings("ignore")

import pennylane as qml
from pennylane import numpy as pnp

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, confusion_matrix,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.utils import resample
from sklearn.inspection import permutation_importance
from sklearn.datasets import load_breast_cancer
from scipy.stats import mode
import scipy.stats as stats

from tqdm.auto import tqdm

PQL_DEVICE = "lightning.qubit"
try:
    _test_dev = qml.device("lightning.gpu", wires=2)
    @qml.qnode(_test_dev)
    def _test_fn():
        qml.Hadamard(0)
        return qml.probs(wires=[0])
    _test_fn()
    PQL_DEVICE = "lightning.gpu"
    print("GPU backend: lightning.gpu (cuQuantum)")
except Exception:
    print("GPU backend unavailable, using lightning.qubit (CPU)")

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
    print(f"XGBoost {xgb.__version__} available")
except ImportError:
    XGB_AVAILABLE = False

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

OUT = "/kaggle/working"
FIG_DIR = os.path.join(OUT, f"figures{SUFFIX}")
TBL_DIR = os.path.join(OUT, f"tables{SUFFIX}")
CKP_DIR = os.path.join(OUT, "checkpoints")
for d in [FIG_DIR, TBL_DIR, CKP_DIR]:
    os.makedirs(d, exist_ok=True)

print(f"PennyLane {qml.__version__}, device: {PQL_DEVICE}")
print(f"Outputs -> {FIG_DIR}, {TBL_DIR}, {CKP_DIR}")

# %% cell 3: Quantum circuit definitions
n_qubits = 8
dev_8 = qml.device(PQL_DEVICE, wires=8)
dev_3 = qml.device(PQL_DEVICE, wires=3)

@qml.qnode(dev_8, interface="autograd")
def angle_kernel_circuit(x1, x2):
    qml.AngleEmbedding(x1, wires=range(8))
    qml.adjoint(qml.AngleEmbedding)(x2, wires=range(8))
    return qml.probs(wires=range(8))

def angle_kernel(x1, x2):
    return angle_kernel_circuit(x1, x2)[0]

@qml.qnode(dev_8, interface="autograd")
def zz_kernel_circuit(x1, x2):
    qml.IQPEmbedding(x1, wires=range(8))
    qml.adjoint(qml.IQPEmbedding)(x2, wires=range(8))
    return qml.probs(wires=range(8))

def zz_kernel(x1, x2):
    return zz_kernel_circuit(x1, x2)[0]

@qml.qnode(dev_3, interface="autograd")
def amplitude_kernel_circuit(x1, x2):
    qml.AmplitudeEmbedding(x1, wires=range(3), normalize=True)
    qml.adjoint(qml.AmplitudeEmbedding)(x2, wires=range(3), normalize=True)
    return qml.probs(wires=range(3))

def amplitude_kernel(x1, x2):
    return amplitude_kernel_circuit(x1, x2)[0]

n_layers = 2
dev_qnn = qml.device(PQL_DEVICE, wires=n_qubits)

@qml.qnode(dev_qnn, interface="autograd")
def qnn_circuit(weights, x):
    qml.AngleEmbedding(x, wires=range(n_qubits))
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return qml.expval(qml.PauliZ(0))

def qnn_cost(weights, X, Y):
    predictions = [qnn_circuit(weights, x) for x in X]
    return pnp.mean((predictions - Y) ** 2)

def qnn_accuracy(weights, X, Y):
    predictions = [np.sign(qnn_circuit(weights, x)) for x in X]
    return accuracy_score(Y, predictions)

print("Quantum circuits defined.")

# %% cell 4: Helper functions
def compute_gram_matrix(X, kernel_func, desc="Gram"):
    """Compute symmetric NxN Gram matrix."""
    n = len(X)
    gram = np.zeros((n, n))
    total = n * (n + 1) // 2
    with tqdm(total=total, desc=desc, leave=False) as pbar:
        for i in range(n):
            for j in range(i, n):
                val = kernel_func(X[i], X[j])
                gram[i, j] = val
                gram[j, i] = val
                pbar.update(1)
    return gram

def get_oof_predictions(model_type, g_matrix, y_target, train_idx, n_splits=5):
    """Inner OOF loop for stacking meta-feature generation."""
    inner_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(train_idx))
    y_sub = y_target[train_idx]
    for itr_pos, ival_pos in inner_cv.split(np.zeros(len(train_idx)), y_sub):
        itr = train_idx[itr_pos]
        ival = train_idx[ival_pos]
        if model_type == "svm":
            m = SVC(kernel="precomputed")
            m.fit(g_matrix[itr][:, itr], y_target[itr])
            oof_preds[ival_pos] = m.predict(g_matrix[ival][:, itr])
        elif model_type == "knn":
            m = KNeighborsClassifier(n_neighbors=5, metric="precomputed")
            m.fit(np.clip(1.0 - g_matrix[itr][:, itr], 0, None), y_target[itr])
            oof_preds[ival_pos] = m.predict(np.clip(1.0 - g_matrix[ival][:, itr], 0, None))
    return oof_preds

def compute_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "acc": accuracy_score(y_true, y_pred),
        "prec": precision_score(y_true, y_pred, zero_division=0),
        "rec": recall_score(y_true, y_pred),
        "spec": tn / (tn + fp) if (tn + fp) > 0 else 0,
    }

def save_fig(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=600, bbox_inches="tight", facecolor="white")
    print(f"  Saved figure: {path}")

def save_table(df, name):
    csv_path = os.path.join(TBL_DIR, f"{name}.csv")
    df.to_csv(csv_path)
    print(f"  Saved table: {csv_path}")

print("Helpers defined.")

# %% cell 5: Data preparation (all 3 datasets)
park_path = "/kaggle/input/datasets/miitdaga/uci-parkinsons-disease-dataset/parkinsons.data"
df_park = pd.read_csv(park_path)
park_feature_names = [c for c in df_park.columns if c not in ["name", "status"]]
X_park_raw = df_park.drop(["name", "status"], axis=1).values
y_park = df_park["status"].values

X_park_std = StandardScaler().fit_transform(X_park_raw)
pca_park = PCA(n_components=8, random_state=42).fit(X_park_std)
X_park_pca = pca_park.transform(X_park_std)
X_park = MinMaxScaler(feature_range=(0, np.pi)).fit_transform(X_park_pca)
print(f"Parkinson's: {X_park.shape}, PCA variance: {pca_park.explained_variance_ratio_.sum():.4f}")
print(f"  Class balance: {np.bincount(y_park)}")

bc_data = load_breast_cancer()
bc_feature_names = list(bc_data.feature_names)
X_bc_raw = bc_data.data
y_bc = bc_data.target

X_bc_std = StandardScaler().fit_transform(X_bc_raw)
pca_bc = PCA(n_components=8, random_state=42).fit(X_bc_std)
X_bc_pca = pca_bc.transform(X_bc_std)
X_bc = MinMaxScaler(feature_range=(0, np.pi)).fit_transform(X_bc_pca)
print(f"Breast Cancer: {X_bc.shape}, PCA variance: {pca_bc.explained_variance_ratio_.sum():.4f}")
print(f"  Class balance: {np.bincount(y_bc)}")

diab_path = "/kaggle/input/datasets/organizations/uciml/pima-indians-diabetes-database/diabetes.csv"
df_diab = pd.read_csv(diab_path)
diab_feature_names = list(df_diab.columns[:-1])
X_diab_raw = df_diab.drop("Outcome", axis=1).values
y_diab = df_diab["Outcome"].values

X_diab_std = StandardScaler().fit_transform(X_diab_raw)
pca_diab = PCA(n_components=8, random_state=42).fit(X_diab_std)
X_diab_pca = pca_diab.transform(X_diab_std)
X_diab = MinMaxScaler(feature_range=(0, np.pi)).fit_transform(X_diab_pca)
print(f"Diabetes: {X_diab.shape}, PCA variance: {pca_diab.explained_variance_ratio_.sum():.4f}")
print(f"  Class balance: {np.bincount(y_diab)}")

DATASETS = {
    "Parkinson's": {
        "X": X_park, "y": y_park, "pca": pca_park,
        "raw_names": park_feature_names, "X_std": X_park_std,
    },
    "Breast Cancer": {
        "X": X_bc, "y": y_bc, "pca": pca_bc,
        "raw_names": bc_feature_names, "X_std": X_bc_std,
    },
    "Diabetes": {
        "X": X_diab, "y": y_diab, "pca": pca_diab,
        "raw_names": diab_feature_names, "X_std": X_diab_std,
    },
}
print("\nAll 3 datasets prepared.")

# %% cell 6: Scree plot (NEW - Reviewer 2, Major 2)
print("="*60)
print("NEW EXPERIMENT: Scree Plot — Justifying 8 PCA Components")
print("="*60)

fig, axes = plt.subplots(1, 3, figsize=(20, 5))
scree_data = {}

for idx, (ds_name, ds) in enumerate(DATASETS.items()):
    n_max = ds["X_std"].shape[1]
    pca_full = PCA(n_components=min(n_max, 30), random_state=42).fit(ds["X_std"])
    var_ratio = pca_full.explained_variance_ratio_
    cumvar = np.cumsum(var_ratio)
    n_comp = len(var_ratio)
    scree_data[ds_name] = {"individual": var_ratio, "cumulative": cumvar}

    ax = axes[idx]
    ax2 = ax.twinx()
    ax.bar(range(1, n_comp + 1), var_ratio * 100, alpha=0.6, color="#3498DB",
           label="Individual")
    ax2.plot(range(1, n_comp + 1), cumvar * 100, "o-", color="#E74C3C",
             linewidth=2, markersize=5, label="Cumulative")
    ax2.axhline(y=cumvar[min(7, n_comp - 1)] * 100, color="gray",
                linestyle="--", linewidth=1)
    ax2.axvline(x=8, color="green", linestyle="--", linewidth=1.5,
                label="n=8 cutoff")
    retained = cumvar[min(7, n_comp - 1)] * 100
    ax2.annotate(f"{retained:.1f}%", xy=(8, retained),
                 xytext=(8 + 1, retained - 5), fontsize=10, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="green"))
    ax.set_xlabel("Principal Component", fontsize=11)
    ax.set_ylabel("Individual Variance (%)", fontsize=10)
    ax2.set_ylabel("Cumulative Variance (%)", fontsize=10)
    ax2.set_ylim(0, 105)
    ax.set_title(ds_name, fontsize=13, fontweight="bold")
    if idx == 0:
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="center right")

plt.suptitle("Scree Plot: Explained Variance by PCA Components",
             fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
save_fig(fig, "scree_plot.png")
plt.show()
print("Scree plot complete.")

# %% cell 7: Kernel PCA / UMAP comparison (NEW - Reviewer 2, Major 2)
print("="*60)
print("NEW EXPERIMENT: Nonlinear DR Comparison (Kernel PCA, UMAP)")
print("="*60)

dr_results = []
cv_dr = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for ds_name, ds in DATASETS.items():
    X_std = ds["X_std"]
    y_ds = ds["y"]
    X_pca8 = ds["X"]

    accs_linear = []
    for tr, te in cv_dr.split(X_pca8, y_ds):
        accs_linear.append(accuracy_score(
            y_ds[te], SVC(kernel="rbf").fit(X_pca8[tr], y_ds[tr]).predict(X_pca8[te])))
    dr_results.append({"Dataset": ds_name, "Method": "Linear PCA (n=8)",
                        "Accuracy": np.mean(accs_linear),
                        "Std": np.std(accs_linear)})

    for n_comp in [7, 8]:
        kpca = KernelPCA(n_components=n_comp, kernel="rbf", random_state=42)
        X_kpca = kpca.fit_transform(X_std)
        X_kpca_sc = MinMaxScaler(feature_range=(0, np.pi)).fit_transform(X_kpca)
        accs_k = []
        for tr, te in cv_dr.split(X_kpca_sc, y_ds):
            accs_k.append(accuracy_score(
                y_ds[te], SVC(kernel="rbf").fit(X_kpca_sc[tr], y_ds[tr]).predict(X_kpca_sc[te])))
        dr_results.append({"Dataset": ds_name, "Method": f"Kernel PCA (n={n_comp})",
                            "Accuracy": np.mean(accs_k), "Std": np.std(accs_k)})

    if UMAP_AVAILABLE:
        for n_comp in [7, 8]:
            um = umap.UMAP(n_components=n_comp, random_state=42, n_neighbors=15)
            X_umap = um.fit_transform(X_std)
            X_umap_sc = MinMaxScaler(feature_range=(0, np.pi)).fit_transform(X_umap)
            accs_u = []
            for tr, te in cv_dr.split(X_umap_sc, y_ds):
                accs_u.append(accuracy_score(
                    y_ds[te], SVC(kernel="rbf").fit(X_umap_sc[tr], y_ds[tr]).predict(X_umap_sc[te])))
            dr_results.append({"Dataset": ds_name, "Method": f"UMAP (n={n_comp})",
                                "Accuracy": np.mean(accs_u), "Std": np.std(accs_u)})

df_dr = pd.DataFrame(dr_results)
df_dr["Accuracy"] = df_dr["Accuracy"].round(4)
df_dr["Std"] = df_dr["Std"].round(4)

for ds_name in DATASETS:
    print(f"\n=== {ds_name}: Dimensionality Reduction Comparison ===")
    sub = df_dr[df_dr["Dataset"] == ds_name][["Method", "Accuracy", "Std"]]
    print(sub.to_string(index=False))

save_table(df_dr, "dr_comparison")
print("\nNonlinear DR comparison complete.")

# %% cell 8: Compute all Gram matrices (with cache)
print("="*60)
print("Quantum Gram Matrices (3 encodings x 3 datasets)")
print("="*60)
print("These are fully deterministic — cached .npy files are reused if present.")

GRAMS = {}
KERNEL_MAP = {"angle": angle_kernel, "zz": zz_kernel, "amp": amplitude_kernel}

for ds_name, ds in DATASETS.items():
    X = ds["X"]
    ds_key = ds_name.lower().replace(" ", "_").replace("'", "")
    GRAMS[ds_name] = {}
    all_cached = True

    for enc_name in ["angle", "zz", "amp"]:
        cache_path = os.path.join(CKP_DIR, f"gram_{enc_name}_{ds_key}.npy")
        if os.path.exists(cache_path):
            GRAMS[ds_name][enc_name] = np.load(cache_path)
            print(f"  [CACHED] {ds_name} {enc_name} <- {cache_path}")
        else:
            all_cached = False
            print(f"  [COMPUTING] {ds_name} {enc_name} (N={len(X)})...")
            t0 = time.time()
            g = compute_gram_matrix(X, KERNEL_MAP[enc_name], f"{ds_name} {enc_name}")
            GRAMS[ds_name][enc_name] = g
            np.save(cache_path, g)
            print(f"    Done in {time.time()-t0:.1f}s, saved to {cache_path}")

    if all_cached:
        print(f"  {ds_name}: all 3 loaded from cache (0 compute)")

print("\nAll Gram matrices ready.")

# %% cell 9: VQC standalone learning curves (all 3 datasets)
print("="*60)
print("VQC Standalone Training — Learning Curves")
print("="*60)

vqc_curves = {}

for ds_name, ds in DATASETS.items():
    print(f"\n--- Training VQC for {ds_name} ({VQC_EPOCHS_STANDALONE} epochs) ---")
    X_all, y_all = ds["X"], ds["y"]
    y_vqc = y_all * 2 - 1

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, y_vqc, test_size=0.2, random_state=42, stratify=y_vqc)
    X_tr_p = pnp.array(X_tr, requires_grad=False)
    Y_tr_p = pnp.array(y_tr, requires_grad=False)
    X_te_p = pnp.array(X_te, requires_grad=False)
    Y_te_p = pnp.array(y_te, requires_grad=False)

    pnp.random.seed(42)
    shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
    weights = pnp.random.random(size=shape, requires_grad=True)
    opt = qml.NesterovMomentumOptimizer(stepsize=0.1)

    train_accs, test_accs = [], []
    progress = tqdm(range(VQC_EPOCHS_STANDALONE), desc=f"VQC {ds_name}")
    for ep in progress:
        b_idx = np.random.randint(0, len(X_tr_p), 16)
        weights, cost = opt.step_and_cost(
            lambda w: qnn_cost(w, X_tr_p[b_idx], Y_tr_p[b_idx]), weights)
        tr_acc = qnn_accuracy(weights, X_tr_p, Y_tr_p)
        te_acc = qnn_accuracy(weights, X_te_p, Y_te_p)
        train_accs.append(float(tr_acc))
        test_accs.append(float(te_acc))
        progress.set_postfix(train=f"{tr_acc:.3f}", test=f"{te_acc:.3f}")

    vqc_curves[ds_name] = {"train": train_accs, "test": test_accs}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for idx, (ds_name, curves) in enumerate(vqc_curves.items()):
    ax = axes[idx]
    ax.plot(curves["train"], label="Train Accuracy", color="blue", linewidth=2)
    ax.plot(curves["test"], label="Test Accuracy", color="orange", linewidth=2, linestyle="--")
    ax.set_title(ds_name, fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.set_ylim(0, 1.05)

plt.suptitle("VQC Learning Curves: Barren Plateau Confirmation",
             fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
save_fig(fig, "vqc_learning_curves.png")
plt.show()

for ds_name, curves in vqc_curves.items():
    fig_single, ax = plt.subplots(figsize=(6, 4))
    ax.plot(curves["train"], label="Train Accuracy", color="blue", linewidth=2)
    ax.plot(curves["test"], label="Test Accuracy", color="orange", linewidth=2, linestyle="--")
    ax.set_title(f"VQC Learning Curve — {ds_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.legend(); ax.grid(True, linestyle=":", alpha=0.7); ax.set_ylim(0, 1.05)
    fname = f"VQC_{ds_name.lower().replace(' ', '_').replace(chr(39), '')}.png"
    save_fig(fig_single, fname)
    plt.close(fig_single)

# %% cell 10: 2D Decision boundaries (all 3 datasets)
print("="*60)
print("Decision Boundaries: Classical RBF vs Quantum ZZ (2D PCA)")
print("="*60)

dev_2q = qml.device(PQL_DEVICE, wires=2)

@qml.qnode(dev_2q, interface="autograd")
def zz_kernel_2d_circuit(x1, x2):
    qml.IQPEmbedding(x1, wires=range(2))
    qml.adjoint(qml.IQPEmbedding)(x2, wires=range(2))
    return qml.probs(wires=range(2))

def zz_kernel_2d(x1, x2):
    return zz_kernel_2d_circuit(x1, x2)[0]

for ds_name, ds in DATASETS.items():
    print(f"\n--- {ds_name} ---")
    X_std, y_ds = ds["X_std"], ds["y"]
    pca_2d = PCA(n_components=2, random_state=42)
    X_2d = pca_2d.fit_transform(X_std)
    X_2d_sc = MinMaxScaler(feature_range=(0, np.pi)).fit_transform(X_2d)

    gram_2d = compute_gram_matrix(X_2d_sc, zz_kernel_2d, f"2D Gram {ds_name}")

    svm_cl = SVC(kernel="rbf").fit(X_2d_sc, y_ds)
    svm_qm = SVC(kernel="precomputed").fit(gram_2d, y_ds)

    res = 25
    xx, yy = np.meshgrid(np.linspace(0, np.pi, res), np.linspace(0, np.pi, res))
    grid = np.c_[xx.ravel(), yy.ravel()]

    grid_gram = np.zeros((len(grid), len(X_2d_sc)))
    for i in range(len(grid)):
        for j in range(len(X_2d_sc)):
            grid_gram[i, j] = zz_kernel_2d(grid[i], X_2d_sc[j])

    Z_cl = svm_cl.predict(grid).reshape(xx.shape)
    Z_qm = svm_qm.predict(grid_gram).reshape(xx.shape)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    cmap = plt.cm.coolwarm
    for ax, Z, title in zip(axes, [Z_cl, Z_qm],
                             ["Classical SVM (RBF)", "Quantum SVM (ZZ-Feature Map)"]):
        ax.contourf(xx, yy, Z, alpha=0.4, cmap=cmap)
        ax.scatter(X_2d_sc[:, 0], X_2d_sc[:, 1], c=y_ds, cmap=cmap, edgecolors="k", s=20)
        ax.set_title(title, fontsize=13, fontweight="bold")
    plt.suptitle(f"{ds_name}: Classical vs Quantum Decision Boundaries",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    fname = f"decision_boundary_{ds_name.lower().replace(' ', '_').replace(chr(39), '')}.png"
    save_fig(fig, fname)
    plt.show()

# %% cell 11: Master evaluation — all 10 quantum + classical RBF models
print("="*60)
print("MASTER EVALUATION: 10 Models x 3 Datasets (Parametric + Bootstrap)")
print("="*60)

def run_parametric(X_class, gram_zz, gram_amp, gram_angle, y, ds_name):
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    model_names = [
        "Classical RBF", "QSVM (Angle)", "QSVM (ZZ Map)", "QSVM (Amplitude)",
        "QKNN (Angle)", "QKNN (ZZ Map)", "QKNN (Amplitude)", "Pure QNN (VQC)",
        "Hard Voting Ensemble", "OOF Stacking Ensemble",
    ]
    folds = {m: {"acc": [], "prec": [], "rec": [], "spec": []} for m in model_names}

    for train_idx, test_idx in tqdm(cv.split(X_class, y), total=10, desc=f"Param {ds_name}"):
        y_tr, y_te = y[train_idx], y[test_idx]
        g_zz_tr  = gram_zz[train_idx][:, train_idx];   g_zz_te  = gram_zz[test_idx][:, train_idx]
        g_amp_tr = gram_amp[train_idx][:, train_idx];   g_amp_te = gram_amp[test_idx][:, train_idx]
        g_ang_tr = gram_angle[train_idx][:, train_idx]; g_ang_te = gram_angle[test_idx][:, train_idx]

        p_c       = SVC(kernel="rbf").fit(X_class[train_idx], y_tr).predict(X_class[test_idx])
        p_svm_ang = SVC(kernel="precomputed").fit(g_ang_tr, y_tr).predict(g_ang_te)
        p_svm_zz  = SVC(kernel="precomputed").fit(g_zz_tr, y_tr).predict(g_zz_te)
        p_svm_amp = SVC(kernel="precomputed").fit(g_amp_tr, y_tr).predict(g_amp_te)
        p_knn_ang = KNeighborsClassifier(n_neighbors=5, metric="precomputed").fit(
            np.clip(1-g_ang_tr,0,None), y_tr).predict(np.clip(1-g_ang_te,0,None))
        p_knn_zz  = KNeighborsClassifier(n_neighbors=5, metric="precomputed").fit(
            np.clip(1-g_zz_tr,0,None), y_tr).predict(np.clip(1-g_zz_te,0,None))
        p_knn_amp = KNeighborsClassifier(n_neighbors=5, metric="precomputed").fit(
            np.clip(1-g_amp_tr,0,None), y_tr).predict(np.clip(1-g_amp_te,0,None))

        p_hv = mode(np.vstack((p_svm_zz, p_knn_amp, p_svm_ang)), axis=0, keepdims=False).mode

        oof_sa = get_oof_predictions("svm", gram_amp, y, train_idx)
        oof_ka = get_oof_predictions("knn", gram_amp, y, train_idx)
        oof_sz = get_oof_predictions("svm", gram_zz, y, train_idx)
        oof_kn = get_oof_predictions("knn", gram_angle, y, train_idx)
        X_meta_tr = np.column_stack((oof_sa, oof_ka, oof_sz, oof_kn))
        X_meta_te = np.column_stack((p_svm_amp, p_knn_amp, p_svm_zz, p_knn_ang))
        meta = LogisticRegression(random_state=42, class_weight="balanced", max_iter=1000)
        meta.fit(X_meta_tr, y_tr)
        p_stack = meta.predict(X_meta_te)

        w = pnp.random.random(size=(2, 8, 3), requires_grad=True)
        opt = qml.NesterovMomentumOptimizer(stepsize=0.1)
        for _ in range(VQC_EPOCHS_INFOLD):
            bi = np.random.randint(0, len(train_idx), 16)
            w, _ = opt.step_and_cost(
                lambda ww: qnn_cost(ww, pnp.array(X_class[train_idx][bi], requires_grad=False),
                                    pnp.array(y_tr[bi]*2-1, requires_grad=False)), w)
        p_vqc = (np.array([np.sign(qnn_circuit(w, x))
                           for x in pnp.array(X_class[test_idx], requires_grad=False)]) + 1) // 2

        preds_dict = {
            "Classical RBF": p_c, "QSVM (Angle)": p_svm_ang,
            "QSVM (ZZ Map)": p_svm_zz, "QSVM (Amplitude)": p_svm_amp,
            "QKNN (Angle)": p_knn_ang, "QKNN (ZZ Map)": p_knn_zz,
            "QKNN (Amplitude)": p_knn_amp, "Pure QNN (VQC)": p_vqc,
            "Hard Voting Ensemble": p_hv, "OOF Stacking Ensemble": p_stack,
        }
        for m_name, preds in preds_dict.items():
            m = compute_metrics(y_te, preds)
            for k in ["acc", "prec", "rec", "spec"]:
                folds[m_name][k].append(m[k])

    rows = []
    base = {k: np.array(folds["Classical RBF"][k]) for k in ["acc","prec","rec","spec"]}
    for m_name in model_names:
        row = {"Model": m_name}
        for metric, label in zip(["acc","prec","rec","spec"],
                                  ["Accuracy","Precision","Recall","Specificity"]):
            arr = np.array(folds[m_name][metric])
            ci = stats.t.interval(0.95, len(arr)-1, loc=np.mean(arr), scale=stats.sem(arr))
            if np.isnan(ci[1]):
                ci = (np.mean(arr), np.mean(arr))
            pv = "-"
            if m_name != "Classical RBF":
                _, p = stats.ttest_rel(arr, base[metric])
                pv = f"{p:.4f}"
            row[f"{label} (Mean +/- 95% CI)"] = f"{np.mean(arr):.3f} +/- {(ci[1]-np.mean(arr)):.3f}"
            row[f"{label} p-val"] = pv
        rows.append(row)
    return pd.DataFrame(rows).set_index("Model"), folds

def run_bootstrap(X_class, gram_zz, gram_amp, gram_angle, y, ds_name):
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    model_names = [
        "Classical RBF", "QSVM (Angle)", "QSVM (ZZ Map)", "QSVM (Amplitude)",
        "QKNN (Angle)", "QKNN (ZZ Map)", "QKNN (Amplitude)", "Pure QNN (VQC)",
        "Hard Voting Ensemble", "OOF Stacking Ensemble",
    ]
    oof = {m: np.zeros(len(y)) for m in model_names}

    for train_idx, test_idx in tqdm(cv.split(X_class, y), total=10, desc=f"Boot OOF {ds_name}"):
        y_tr = y[train_idx]
        g_zz_tr  = gram_zz[train_idx][:, train_idx];   g_zz_te  = gram_zz[test_idx][:, train_idx]
        g_amp_tr = gram_amp[train_idx][:, train_idx];   g_amp_te = gram_amp[test_idx][:, train_idx]
        g_ang_tr = gram_angle[train_idx][:, train_idx]; g_ang_te = gram_angle[test_idx][:, train_idx]

        oof["Classical RBF"][test_idx]    = SVC(kernel="rbf").fit(X_class[train_idx], y_tr).predict(X_class[test_idx])
        oof["QSVM (Angle)"][test_idx]     = SVC(kernel="precomputed").fit(g_ang_tr, y_tr).predict(g_ang_te)
        oof["QSVM (ZZ Map)"][test_idx]    = SVC(kernel="precomputed").fit(g_zz_tr, y_tr).predict(g_zz_te)
        oof["QSVM (Amplitude)"][test_idx] = SVC(kernel="precomputed").fit(g_amp_tr, y_tr).predict(g_amp_te)
        oof["QKNN (Angle)"][test_idx]     = KNeighborsClassifier(n_neighbors=5, metric="precomputed").fit(
            np.clip(1-g_ang_tr,0,None), y_tr).predict(np.clip(1-g_ang_te,0,None))
        oof["QKNN (ZZ Map)"][test_idx]    = KNeighborsClassifier(n_neighbors=5, metric="precomputed").fit(
            np.clip(1-g_zz_tr,0,None), y_tr).predict(np.clip(1-g_zz_te,0,None))
        oof["QKNN (Amplitude)"][test_idx] = KNeighborsClassifier(n_neighbors=5, metric="precomputed").fit(
            np.clip(1-g_amp_tr,0,None), y_tr).predict(np.clip(1-g_amp_te,0,None))

        oof["Hard Voting Ensemble"][test_idx] = mode(np.vstack((
            oof["QSVM (ZZ Map)"][test_idx], oof["QKNN (Amplitude)"][test_idx],
            oof["QSVM (Angle)"][test_idx])), axis=0, keepdims=False).mode

        meta_tr = np.column_stack((
            get_oof_predictions("svm", gram_amp, y, train_idx),
            get_oof_predictions("knn", gram_amp, y, train_idx),
            get_oof_predictions("svm", gram_zz, y, train_idx),
            get_oof_predictions("knn", gram_angle, y, train_idx)))
        meta_te = np.column_stack((
            oof["QSVM (Amplitude)"][test_idx], oof["QKNN (Amplitude)"][test_idx],
            oof["QSVM (ZZ Map)"][test_idx], oof["QKNN (Angle)"][test_idx]))
        ml = LogisticRegression(random_state=42, class_weight="balanced", max_iter=1000)
        ml.fit(meta_tr, y_tr)
        oof["OOF Stacking Ensemble"][test_idx] = ml.predict(meta_te)

        w = pnp.random.random(size=(2, 8, 3), requires_grad=True)
        opt = qml.NesterovMomentumOptimizer(stepsize=0.1)
        for _ in range(VQC_EPOCHS_INFOLD):
            bi = np.random.randint(0, len(train_idx), 16)
            w, _ = opt.step_and_cost(
                lambda ww: qnn_cost(ww, pnp.array(X_class[train_idx][bi], requires_grad=False),
                                    pnp.array(y_tr[bi]*2-1, requires_grad=False)), w)
        oof["Pure QNN (VQC)"][test_idx] = (np.array([
            np.sign(qnn_circuit(w, x))
            for x in pnp.array(X_class[test_idx], requires_grad=False)]) + 1) // 2

    boot = {m: {"acc":[], "prec":[], "rec":[], "spec":[]} for m in model_names}
    np.random.seed(42)
    for _ in range(N_BOOTSTRAPS):
        idx = resample(np.arange(len(y)), stratify=y)
        yt = y[idx]
        for m in model_names:
            m_dict = compute_metrics(yt, oof[m][idx])
            for k in m_dict:
                boot[m][k].append(m_dict[k])

    rows = []
    base = {k: np.array(boot["Classical RBF"][k]) for k in ["acc","prec","rec","spec"]}
    for m_name in model_names:
        row = {"Model": m_name}
        for metric, label in zip(["acc","prec","rec","spec"],
                                  ["Accuracy","Precision","Recall","Specificity"]):
            arr = np.array(boot[m_name][metric])
            ci = (np.percentile(arr, 2.5), np.percentile(arr, 97.5))
            pv = "-"
            if m_name != "Classical RBF":
                diff = arr - base[metric]
                p = min(1.0, 2 * min(np.mean(diff <= 0), np.mean(diff >= 0)))
                pv = "<0.001" if p == 0 else f"{p:.4f}"
            row[f"{label} [95% CI]"] = f"{np.mean(arr):.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"
            row[f"{label} p-val"] = pv
        rows.append(row)
    return pd.DataFrame(rows).set_index("Model"), oof

PARAM_RESULTS, PARAM_FOLDS = {}, {}
BOOT_RESULTS, OOF_PREDS = {}, {}

for ds_name in DATASETS:
    g = GRAMS[ds_name]
    X_class, y_ds = DATASETS[ds_name]["X"], DATASETS[ds_name]["y"]
    print(f"\n{'='*50}\n{ds_name.upper()}\n{'='*50}")

    df_p, folds = run_parametric(X_class, g["zz"], g["amp"], g["angle"], y_ds, ds_name)
    PARAM_RESULTS[ds_name] = df_p
    PARAM_FOLDS[ds_name] = folds
    print(f"\n=== {ds_name}: PARAMETRIC ===")
    print(df_p.to_string())
    save_table(df_p, f"parametric_{ds_name.lower().replace(' ','_').replace(chr(39),'')}")

    df_b, oof = run_bootstrap(X_class, g["zz"], g["amp"], g["angle"], y_ds, ds_name)
    BOOT_RESULTS[ds_name] = df_b
    OOF_PREDS[ds_name] = oof
    print(f"\n=== {ds_name}: BOOTSTRAP ===")
    print(df_b.to_string())
    save_table(df_b, f"bootstrap_{ds_name.lower().replace(' ','_').replace(chr(39),'')}")

print("\nMaster evaluation complete.")

# %% cell 12: Classical ensemble baselines (NEW - Reviewer 1, Comment 4)
print("="*60)
print("NEW EXPERIMENT: Classical Ensemble Baselines")
print("="*60)

CLASSICAL_MODELS = {}
if XGB_AVAILABLE:
    try:
        CLASSICAL_MODELS["XGBoost"] = lambda: xgb.XGBClassifier(
            n_estimators=100, max_depth=4, random_state=42,
            eval_metric="logloss", device="cuda", verbosity=0)
    except Exception:
        CLASSICAL_MODELS["XGBoost"] = lambda: xgb.XGBClassifier(
            n_estimators=100, max_depth=4, random_state=42,
            eval_metric="logloss", verbosity=0)

CLASSICAL_MODELS["Random Forest"] = lambda: RandomForestClassifier(
    n_estimators=100, max_depth=None, random_state=42)

def make_classical_stacking():
    estimators = [
        ("svm_rbf", SVC(kernel="rbf", probability=True, random_state=42)),
        ("knn_5", KNeighborsClassifier(n_neighbors=5)),
    ]
    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(random_state=42, class_weight="balanced"),
        cv=5, passthrough=False)

CLASSICAL_MODELS["Classical Stacking (SVM+KNN)"] = make_classical_stacking

CL_BOOT_RESULTS = {}

for ds_name in DATASETS:
    print(f"\n--- {ds_name} ---")
    X_class, y_ds = DATASETS[ds_name]["X"], DATASETS[ds_name]["y"]
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    cl_oof = {}
    for cl_name, cl_factory in CLASSICAL_MODELS.items():
        oof_pred = np.zeros(len(y_ds))
        for tr, te in cv.split(X_class, y_ds):
            model = cl_factory()
            model.fit(X_class[tr], y_ds[tr])
            oof_pred[te] = model.predict(X_class[te])
        cl_oof[cl_name] = oof_pred
        acc = accuracy_score(y_ds, oof_pred)
        print(f"  {cl_name}: Acc={acc:.4f}")

    rows = []
    np.random.seed(42)
    boot_cl = {m: {"acc":[], "prec":[], "rec":[], "spec":[]} for m in CLASSICAL_MODELS}
    for _ in range(N_BOOTSTRAPS):
        idx = resample(np.arange(len(y_ds)), stratify=y_ds)
        yt = y_ds[idx]
        for m in CLASSICAL_MODELS:
            md = compute_metrics(yt, cl_oof[m][idx])
            for k in md:
                boot_cl[m][k].append(md[k])

    for m_name in CLASSICAL_MODELS:
        row = {"Model": m_name}
        for metric, label in zip(["acc","prec","rec","spec"],
                                  ["Accuracy","Precision","Recall","Specificity"]):
            arr = np.array(boot_cl[m_name][metric])
            ci = (np.percentile(arr, 2.5), np.percentile(arr, 97.5))
            row[f"{label} [95% CI]"] = f"{np.mean(arr):.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"
        rows.append(row)

    df_cl = pd.DataFrame(rows).set_index("Model")
    CL_BOOT_RESULTS[ds_name] = df_cl
    print(f"\n=== {ds_name}: Classical Baselines Bootstrap ===")
    print(df_cl.to_string())
    save_table(df_cl, f"classical_baselines_{ds_name.lower().replace(' ','_').replace(chr(39),'')}")

print("\nClassical baselines complete.")

# %% cell 13: SHAP analysis
print("="*60)
print("SHAP Analysis of OOF Stacking Meta-Learner")
print("="*60)
import shap

BASE_MODEL_NAMES = ["QSVM (Amplitude)", "QKNN (Amplitude)", "QSVM (ZZ Map)", "QKNN (Angle)"]

def build_shap_analysis(X_class, gram_zz, gram_amp, gram_angle, y):
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    oof_sa, oof_ka, oof_sz, oof_kn = [np.zeros(len(y)) for _ in range(4)]
    for tr, te in cv.split(X_class, y):
        y_tr = y[tr]
        oof_sa[te] = SVC(kernel="precomputed").fit(gram_amp[tr][:,tr], y_tr).predict(gram_amp[te][:,tr])
        oof_ka[te] = KNeighborsClassifier(n_neighbors=5, metric="precomputed").fit(
            np.clip(1-gram_amp[tr][:,tr],0,None), y_tr).predict(np.clip(1-gram_amp[te][:,tr],0,None))
        oof_sz[te] = SVC(kernel="precomputed").fit(gram_zz[tr][:,tr], y_tr).predict(gram_zz[te][:,tr])
        oof_kn[te] = KNeighborsClassifier(n_neighbors=5, metric="precomputed").fit(
            np.clip(1-gram_angle[tr][:,tr],0,None), y_tr).predict(np.clip(1-gram_angle[te][:,tr],0,None))

    X_meta = np.column_stack((oof_sa, oof_ka, oof_sz, oof_kn))
    X_meta_df = pd.DataFrame(X_meta, columns=BASE_MODEL_NAMES)
    ml = LogisticRegression(random_state=42, class_weight="balanced").fit(X_meta, y)
    explainer = shap.LinearExplainer(ml, X_meta_df)
    sv = explainer.shap_values(X_meta_df)
    return X_meta_df, ml, sv

shap_res = {}
for ds_name in DATASETS:
    g = GRAMS[ds_name]
    X_class, y_ds = DATASETS[ds_name]["X"], DATASETS[ds_name]["y"]
    print(f"  Computing SHAP for {ds_name}...")
    xm, ml, sv = build_shap_analysis(X_class, g["zz"], g["amp"], g["angle"], y_ds)
    shap_res[ds_name] = {"X_meta": xm, "model": ml, "shap_values": sv}

fig, axes = plt.subplots(1, 3, figsize=(24, 6))
for idx, (ds_name, res) in enumerate(shap_res.items()):
    plt.sca(axes[idx])
    shap.summary_plot(res["shap_values"], res["X_meta"],
                      feature_names=BASE_MODEL_NAMES, show=False, plot_size=None)
    axes[idx].set_title(ds_name, fontsize=14, fontweight="bold")
plt.suptitle("SHAP Beeswarm: Quantum Kernel Trust by Dataset",
             fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
save_fig(fig, "shap_beeswarm_all_datasets.png")
plt.show()

coeff_rows = []
for ds_name, res in shap_res.items():
    coeffs = res["model"].coef_[0]
    mean_shap = np.abs(res["shap_values"]).mean(axis=0)
    for i, bn in enumerate(BASE_MODEL_NAMES):
        coeff_rows.append({"Dataset": ds_name, "Base Model": bn,
                           "Coefficient": coeffs[i], "Mean |SHAP|": mean_shap[i]})
df_shap = pd.DataFrame(coeff_rows)
print("\n=== Meta-Learner Coefficients & SHAP ===")
for ds in DATASETS:
    print(f"\n{ds}:")
    print(df_shap[df_shap["Dataset"]==ds][["Base Model","Coefficient","Mean |SHAP|"]].to_string(index=False))
save_table(df_shap, "shap_coefficients")

fig, axes = plt.subplots(1, 2, figsize=(20, 6))
for ax_idx, col, title in [(0, "Mean |SHAP|", "Mean |SHAP| by Dataset"),
                             (1, "Coefficient", "Meta-Learner Coefficients")]:
    pivot = df_shap.pivot(index="Base Model", columns="Dataset", values=col)
    pivot.plot(kind="bar", ax=axes[ax_idx], edgecolor="black", alpha=0.85)
    axes[ax_idx].set_title(title, fontsize=13, fontweight="bold")
    axes[ax_idx].set_ylabel(col)
    axes[ax_idx].tick_params(axis="x", rotation=15)
    if col == "Coefficient":
        axes[ax_idx].axhline(y=0, color="black", linewidth=0.8, linestyle="--")
plt.tight_layout()
save_fig(fig, "shap_meta_coefficients.png")
plt.show()

# %% cell 14: LIME + Permutation Importance + XAI agreement
print("="*60)
print("LIME + Permutation Importance + XAI Agreement")
print("="*60)
import lime, lime.lime_tabular

meta_data = {}
for ds_name in DATASETS:
    g = GRAMS[ds_name]
    X_class, y_ds = DATASETS[ds_name]["X"], DATASETS[ds_name]["y"]
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    oof_sa, oof_ka, oof_sz, oof_kn = [np.zeros(len(y_ds)) for _ in range(4)]
    for tr, te in cv.split(X_class, y_ds):
        y_tr = y_ds[tr]
        oof_sa[te] = SVC(kernel="precomputed").fit(g["amp"][tr][:,tr], y_tr).predict(g["amp"][te][:,tr])
        oof_ka[te] = KNeighborsClassifier(n_neighbors=5, metric="precomputed").fit(
            np.clip(1-g["amp"][tr][:,tr],0,None), y_tr).predict(np.clip(1-g["amp"][te][:,tr],0,None))
        oof_sz[te] = SVC(kernel="precomputed").fit(g["zz"][tr][:,tr], y_tr).predict(g["zz"][te][:,tr])
        oof_kn[te] = KNeighborsClassifier(n_neighbors=5, metric="precomputed").fit(
            np.clip(1-g["angle"][tr][:,tr],0,None), y_tr).predict(np.clip(1-g["angle"][te][:,tr],0,None))
    X_meta = np.column_stack((oof_sa, oof_ka, oof_sz, oof_kn))
    X_meta_df = pd.DataFrame(X_meta, columns=BASE_MODEL_NAMES)
    ml = LogisticRegression(random_state=42, class_weight="balanced").fit(X_meta, y_ds)
    meta_data[ds_name] = {"X_meta": X_meta_df, "model": ml, "y": y_ds}

fig, axes = plt.subplots(1, 3, figsize=(22, 6))
perm_results_all = {}
for idx, (ds_name, data) in enumerate(meta_data.items()):
    perm = permutation_importance(data["model"], data["X_meta"], data["y"],
                                  n_repeats=100, random_state=42, scoring="accuracy")
    perm_results_all[ds_name] = perm
    si = perm.importances_mean.argsort()[::-1]
    ax = axes[idx]
    ax.boxplot([perm.importances[i] for i in si], vert=True,
               labels=[BASE_MODEL_NAMES[i] for i in si], patch_artist=True,
               boxprops=dict(facecolor="#4ECDC4", alpha=0.7),
               medianprops=dict(color="red", linewidth=2))
    ax.set_title(ds_name, fontsize=13, fontweight="bold")
    ax.set_ylabel("Accuracy Drop"); ax.tick_params(axis="x", rotation=25)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
plt.suptitle("Permutation Feature Importance", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
save_fig(fig, "xai_permutation_importance.png")
plt.show()

for ds_name, data in meta_data.items():
    X_np = data["X_meta"].values
    y_ds = data["y"]; model = data["model"]
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_np, feature_names=BASE_MODEL_NAMES,
        class_names=["Negative", "Positive"], mode="classification", random_state=42)
    y_pred = model.predict(X_np)
    tp = np.where((y_pred==1)&(y_ds==1))[0]
    tn = np.where((y_pred==0)&(y_ds==0))[0]
    mis = np.where(y_pred!=y_ds)[0]
    samples = {}
    if len(tp)>0: samples["True Positive"] = tp[0]
    if len(tn)>0: samples["True Negative"] = tn[0]
    if len(mis)>0: samples["Misclassified"] = mis[0]
    n_s = len(samples)
    fig, axes = plt.subplots(1, n_s, figsize=(7*n_s, 5))
    if n_s == 1: axes = [axes]
    for ax_i, (stype, sidx) in enumerate(samples.items()):
        exp = explainer.explain_instance(X_np[sidx], model.predict_proba,
                                         num_features=4, num_samples=1000)
        elist = exp.as_list(); elist.reverse()
        feats = [e[0] for e in elist]; wts = [e[1] for e in elist]
        colors = ["#FF6B6B" if w<0 else "#4ECDC4" for w in wts]
        ax = axes[ax_i]
        ax.barh(range(len(feats)), wts, color=colors, edgecolor="black", alpha=0.8)
        ax.set_yticks(range(len(feats))); ax.set_yticklabels(feats, fontsize=9)
        ax.set_xlabel("LIME Weight")
        ax.axvline(x=0, color="black", linewidth=0.8, linestyle="--")
        tl = int(y_ds[sidx]); pl = int(model.predict(X_np[sidx].reshape(1,-1))[0])
        pp = model.predict_proba(X_np[sidx].reshape(1,-1))[0]
        ax.set_title(f"{stype}\n(True={tl}, Pred={pl}, P(1)={pp[1]:.3f})", fontsize=11, fontweight="bold")
    plt.suptitle(f"LIME: {ds_name}", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    fname = f"xai_lime_{ds_name.lower().replace(' ','_').replace(chr(39),'')}.png"
    save_fig(fig, fname)
    plt.show()

print("\n=== XAI Method Agreement ===")
xai_rows = []
for ds_name, data in meta_data.items():
    X_np = data["X_meta"].values; model = data["model"]
    perm = perm_results_all[ds_name]
    perm_rank = np.argsort(perm.importances_mean)[::-1]

    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_np, feature_names=BASE_MODEL_NAMES, class_names=["0","1"],
        mode="classification", random_state=42)
    lime_imp = np.zeros(4)
    si = np.random.RandomState(42).choice(len(X_np), min(LIME_SAMPLES, len(X_np)), replace=False)
    for s in si:
        exp = explainer.explain_instance(X_np[s], model.predict_proba, num_features=4, num_samples=500)
        for fn, w in exp.as_list():
            for fi, bn in enumerate(BASE_MODEL_NAMES):
                if bn in fn: lime_imp[fi] += abs(w); break
    lime_imp /= len(si)
    lime_rank = np.argsort(lime_imp)[::-1]
    coeff_rank = np.argsort(np.abs(model.coef_[0]))[::-1]

    for rank in range(4):
        xai_rows.append({"Dataset": ds_name, "Rank": rank+1,
                          "Permutation": BASE_MODEL_NAMES[perm_rank[rank]],
                          "LIME": BASE_MODEL_NAMES[lime_rank[rank]],
                          "|Coefficient|": BASE_MODEL_NAMES[coeff_rank[rank]]})

df_xai = pd.DataFrame(xai_rows)
for ds in DATASETS:
    print(f"\n{ds}:")
    print(df_xai[df_xai["Dataset"]==ds][["Rank","Permutation","LIME","|Coefficient|"]].to_string(index=False))
save_table(df_xai, "xai_agreement")

# %% cell 15: Circuit specs + ROC/PR + McNemar + Ablation
print("="*60)
print("Circuit Specs | ROC/PR Curves | McNemar's Test | Ablation Study")
print("="*60)
from statsmodels.stats.contingency_tables import mcnemar as mcnemar_test_fn

dummy_8 = np.random.uniform(0, np.pi, 8)
n_timing = 1000

t0 = time.time()
for _ in range(n_timing): angle_kernel(dummy_8, dummy_8)
t_angle = (time.time()-t0)/n_timing*1000

t0 = time.time()
for _ in range(n_timing): zz_kernel(dummy_8, dummy_8)
t_zz = (time.time()-t0)/n_timing*1000

x_amp = np.random.uniform(0, 1, 8)
t0 = time.time()
for _ in range(n_timing): amplitude_kernel(x_amp, x_amp)
t_amp = (time.time()-t0)/n_timing*1000

dw = pnp.random.random(size=(2,8,3), requires_grad=False)
t0 = time.time()
for _ in range(n_timing): qnn_circuit(dw, dummy_8)
t_vqc = (time.time()-t0)/n_timing*1000

timing_df = pd.DataFrame({
    "Encoding": ["Angle (RY)", "ZZ/IQP", "Amplitude", "VQC Forward"],
    "Qubits": [8, 8, 3, 8],
    "Time/Call (ms)": [f"{t_angle:.3f}", f"{t_zz:.3f}", f"{t_amp:.3f}", f"{t_vqc:.3f}"],
    "Gram Calls (N=195)": [19110]*3 + ["N/A"],
    "Est. Gram Time (s)": [f"{t_angle*19110/1000:.1f}", f"{t_zz*19110/1000:.1f}",
                           f"{t_amp*19110/1000:.1f}", "N/A"],
})
print("\n=== Kernel Timing ===")
print(timing_df.to_string(index=False))
save_table(timing_df, "circuit_timing")

print("\nComputing ROC/PR curves...")

def compute_roc_pr(X_class, gram_zz, gram_amp, gram_angle, y, ds_name):
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    mnames = ["Classical RBF", "QSVM (Angle)", "QSVM (ZZ Map)", "QSVM (Amplitude)",
              "QKNN (Angle)", "QKNN (ZZ Map)", "QKNN (Amplitude)", "Pure QNN (VQC)",
              "Hard Voting Ensemble", "OOF Stacking Ensemble"]
    oof_p = {m: np.zeros(len(y)) for m in mnames}
    oof_s = {m: np.zeros(len(y)) for m in mnames}

    for tr, te in tqdm(cv.split(X_class, y), total=10, desc=f"ROC {ds_name}"):
        y_tr = y[tr]
        g_zz_tr, g_zz_te = gram_zz[tr][:,tr], gram_zz[te][:,tr]
        g_amp_tr, g_amp_te = gram_amp[tr][:,tr], gram_amp[te][:,tr]
        g_ang_tr, g_ang_te = gram_angle[tr][:,tr], gram_angle[te][:,tr]

        clf = SVC(kernel="rbf", probability=True, random_state=42).fit(X_class[tr], y_tr)
        oof_p["Classical RBF"][te] = clf.predict(X_class[te])
        oof_s["Classical RBF"][te] = clf.predict_proba(X_class[te])[:,1]

        for name, g_tr, g_te in [("QSVM (Angle)", g_ang_tr, g_ang_te),
                                  ("QSVM (ZZ Map)", g_zz_tr, g_zz_te),
                                  ("QSVM (Amplitude)", g_amp_tr, g_amp_te)]:
            s = SVC(kernel="precomputed").fit(g_tr, y_tr)
            oof_p[name][te] = s.predict(g_te)
            oof_s[name][te] = s.decision_function(g_te)

        for name, g_tr, g_te in [("QKNN (Angle)", g_ang_tr, g_ang_te),
                                  ("QKNN (ZZ Map)", g_zz_tr, g_zz_te),
                                  ("QKNN (Amplitude)", g_amp_tr, g_amp_te)]:
            k = KNeighborsClassifier(n_neighbors=5, metric="precomputed")
            k.fit(np.clip(1-g_tr,0,None), y_tr)
            oof_p[name][te] = k.predict(np.clip(1-g_te,0,None))
            oof_s[name][te] = k.predict_proba(np.clip(1-g_te,0,None))[:,1]

        w = pnp.random.random(size=(2,8,3), requires_grad=True)
        opt = qml.NesterovMomentumOptimizer(stepsize=0.1)
        for _ in range(VQC_EPOCHS_INFOLD):
            bi = np.random.randint(0, len(tr), 16)
            w, _ = opt.step_and_cost(
                lambda ww: qnn_cost(ww, pnp.array(X_class[tr][bi], requires_grad=False),
                                    pnp.array(y_tr[bi]*2-1, requires_grad=False)), w)
        raw = np.array([float(qnn_circuit(w, x)) for x in pnp.array(X_class[te], requires_grad=False)])
        oof_p["Pure QNN (VQC)"][te] = ((np.sign(raw)+1)/2).astype(int)
        oof_s["Pure QNN (VQC)"][te] = (raw+1)/2

        oof_p["Hard Voting Ensemble"][te] = mode(np.vstack((
            oof_p["QSVM (ZZ Map)"][te], oof_p["QKNN (Amplitude)"][te],
            oof_p["QSVM (Angle)"][te])), axis=0, keepdims=False).mode
        oof_s["Hard Voting Ensemble"][te] = (
            oof_s["QSVM (ZZ Map)"][te]+oof_s["QKNN (Amplitude)"][te]+oof_s["QSVM (Angle)"][te])/3

        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        meta_tr = np.zeros((len(tr), 4))
        for itr_p, ival_p in inner_cv.split(np.zeros(len(tr)), y_tr):
            itr, ival = tr[itr_p], tr[ival_p]
            meta_tr[ival_p, 0] = SVC(kernel="precomputed").fit(gram_amp[itr][:,itr], y[itr]).predict(gram_amp[ival][:,itr])
            meta_tr[ival_p, 1] = KNeighborsClassifier(n_neighbors=5, metric="precomputed").fit(
                np.clip(1-gram_amp[itr][:,itr],0,None), y[itr]).predict(np.clip(1-gram_amp[ival][:,itr],0,None))
            meta_tr[ival_p, 2] = SVC(kernel="precomputed").fit(gram_zz[itr][:,itr], y[itr]).predict(gram_zz[ival][:,itr])
            meta_tr[ival_p, 3] = KNeighborsClassifier(n_neighbors=5, metric="precomputed").fit(
                np.clip(1-gram_angle[itr][:,itr],0,None), y[itr]).predict(np.clip(1-gram_angle[ival][:,itr],0,None))
        meta_te = np.column_stack((oof_p["QSVM (Amplitude)"][te], oof_p["QKNN (Amplitude)"][te],
                                   oof_p["QSVM (ZZ Map)"][te], oof_p["QKNN (Angle)"][te]))
        ml = LogisticRegression(random_state=42, class_weight="balanced").fit(meta_tr, y_tr)
        oof_p["OOF Stacking Ensemble"][te] = ml.predict(meta_te)
        oof_s["OOF Stacking Ensemble"][te] = ml.predict_proba(meta_te)[:,1]

    return oof_p, oof_s, mnames

roc_preds, roc_scores = {}, {}
for ds_name in DATASETS:
    g = GRAMS[ds_name]
    p, s, mn = compute_roc_pr(DATASETS[ds_name]["X"], g["zz"], g["amp"], g["angle"],
                               DATASETS[ds_name]["y"], ds_name)
    roc_preds[ds_name] = p
    roc_scores[ds_name] = s

color_map = {
    "Classical RBF": "#2C3E50", "QSVM (Angle)": "#E74C3C",
    "QSVM (ZZ Map)": "#3498DB", "QSVM (Amplitude)": "#E67E22",
    "QKNN (Angle)": "#9B59B6", "QKNN (ZZ Map)": "#1ABC9C",
    "QKNN (Amplitude)": "#F39C12", "Pure QNN (VQC)": "#7F8C8D",
    "Hard Voting Ensemble": "#27AE60", "OOF Stacking Ensemble": "#C0392B",
}

fig, axes = plt.subplots(2, 3, figsize=(24, 14))
auc_rows = []
for ci, ds_name in enumerate(DATASETS):
    y_ds = DATASETS[ds_name]["y"]
    scores = roc_scores[ds_name]
    ax_r = axes[0, ci]; ax_p = axes[1, ci]
    ax_r.plot([0,1],[0,1],"k--",alpha=0.5)
    for mn in scores:
        fpr, tpr, _ = roc_curve(y_ds, scores[mn])
        ra = auc(fpr, tpr)
        pr_arr, re_arr, _ = precision_recall_curve(y_ds, scores[mn])
        pa = average_precision_score(y_ds, scores[mn])
        lw = 3.0 if "Ensemble" in mn else (2.5 if "VQC" in mn else 1.5)
        ls = "-" if "Ensemble" in mn else (":" if "VQC" in mn else "--")
        ax_r.plot(fpr, tpr, color=color_map[mn], linewidth=lw, linestyle=ls, label=f"{mn} ({ra:.3f})")
        ax_p.plot(re_arr, pr_arr, color=color_map[mn], linewidth=lw, linestyle=ls, label=f"{mn} ({pa:.3f})")
        auc_rows.append({"Dataset": ds_name, "Model": mn, "ROC-AUC": round(ra,4), "PR-AUC": round(pa,4)})
    ax_r.set_title(ds_name, fontsize=14, fontweight="bold")
    ax_r.grid(True, alpha=0.3)
    ax_p.set_title(ds_name, fontsize=14, fontweight="bold")
    ax_p.grid(True, alpha=0.3)

handles_leg, labels_leg = axes[0, 0].get_legend_handles_labels()
fig.legend(handles_leg, labels_leg, loc="lower center", ncol=5, fontsize=9,
           bbox_to_anchor=(0.5, -0.06), frameon=True, title="Model (AUC shown for first dataset)",
           title_fontsize=10)
plt.suptitle("ROC & Precision-Recall Curves: All 10 Models", fontsize=16, fontweight="bold", y=1.01)
plt.tight_layout(rect=[0, 0.08, 1, 0.98])
save_fig(fig, "roc_pr_curves_all_10_models.png")
plt.show()

df_auc = pd.DataFrame(auc_rows)
for ds in DATASETS:
    print(f"\n=== {ds}: AUC Scores ===")
    print(df_auc[df_auc["Dataset"]==ds][["Model","ROC-AUC","PR-AUC"]].to_string(index=False))
save_table(df_auc, "auc_scores")

print("\n--- McNemar's Test ---")
def run_mcnemar(y_true, pa, pb, na, nb):
    ca = (pa == y_true).astype(int); cb = (pb == y_true).astype(int)
    n01 = int(np.sum((ca==1)&(cb==0))); n10 = int(np.sum((ca==0)&(cb==1)))
    table = np.array([[int(np.sum((ca==1)&(cb==1))), n01],
                      [n10, int(np.sum((ca==0)&(cb==0)))]])
    res = mcnemar_test_fn(table, exact=(n01+n10<25), correction=(n01+n10>=25))
    sig = "***" if res.pvalue<0.001 else "**" if res.pvalue<0.01 else "*" if res.pvalue<0.05 else "ns"
    return {"Comparison": f"{na} vs {nb}", "n01": n01, "n10": n10,
            "Statistic": f"{res.statistic:.2f}",
            "p-value": "<0.001" if res.pvalue<0.001 else f"{res.pvalue:.4f}", "Sig": sig}

mcn_rows = []
for ds_name in DATASETS:
    y_ds = DATASETS[ds_name]["y"]
    preds = roc_preds[ds_name]
    indiv = [m for m in preds if m not in ["Classical RBF","Pure QNN (VQC)","Hard Voting Ensemble","OOF Stacking Ensemble"]]
    best_q = max(indiv, key=lambda m: accuracy_score(y_ds, preds[m]))
    for na, nb in [("OOF Stacking Ensemble","Classical RBF"),
                    ("OOF Stacking Ensemble", best_q),
                    ("OOF Stacking Ensemble","Pure QNN (VQC)")]:
        r = run_mcnemar(y_ds, preds[na], preds[nb], na, nb)
        r["Dataset"] = ds_name; mcn_rows.append(r)

df_mcn = pd.DataFrame(mcn_rows)
for ds in DATASETS:
    print(f"\n{ds}:")
    print(df_mcn[df_mcn["Dataset"]==ds][["Comparison","n01","n10","Statistic","p-value","Sig"]].to_string(index=False))
save_table(df_mcn, "mcnemar_results")

print("\n--- Ablation Study ---")
def run_ablation(X_class, gram_zz, gram_amp, gram_angle, y, ds_name):
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    builders = {
        "QSVM (Amplitude)": lambda tr, te: SVC(kernel="precomputed").fit(gram_amp[tr][:,tr], y[tr]).predict(gram_amp[te][:,tr]),
        "QKNN (Amplitude)": lambda tr, te: KNeighborsClassifier(n_neighbors=5, metric="precomputed").fit(
            np.clip(1-gram_amp[tr][:,tr],0,None), y[tr]).predict(np.clip(1-gram_amp[te][:,tr],0,None)),
        "QSVM (ZZ Map)": lambda tr, te: SVC(kernel="precomputed").fit(gram_zz[tr][:,tr], y[tr]).predict(gram_zz[te][:,tr]),
        "QKNN (Angle)": lambda tr, te: KNeighborsClassifier(n_neighbors=5, metric="precomputed").fit(
            np.clip(1-gram_angle[tr][:,tr],0,None), y[tr]).predict(np.clip(1-gram_angle[te][:,tr],0,None)),
    }
    all_names = list(builders.keys())
    configs = {"Full Ensemble (4 models)": all_names}
    for rm in all_names: configs[f"Remove {rm}"] = [b for b in all_names if b != rm]
    for s in all_names: configs[f"Only {s}"] = [s]

    results = {}
    for cfg, active in tqdm(configs.items(), desc=f"Ablation {ds_name}"):
        mets = {"acc":[],"prec":[],"rec":[],"spec":[]}
        for tr, te in cv.split(X_class, y):
            y_tr, y_te = y[tr], y[te]
            inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            m_tr = np.zeros((len(tr), len(active)))
            m_te = np.zeros((len(te), len(active)))
            for mi, mn in enumerate(active):
                for itr_p, ival_p in inner.split(np.zeros(len(tr)), y_tr):
                    m_tr[ival_p, mi] = builders[mn](tr[itr_p], tr[ival_p])
                m_te[:, mi] = builders[mn](tr, te)
            if len(active) == 1:
                yp = m_te[:,0].astype(int)
            else:
                lr = LogisticRegression(random_state=42, class_weight="balanced")
                lr.fit(m_tr, y_tr)
                yp = lr.predict(m_te)
            cm = compute_metrics(y_te, yp)
            for k in cm: mets[k].append(cm[k])
        results[cfg] = {k: np.mean(v) for k, v in mets.items()}
    return results

abl_all = {}
for ds_name in DATASETS:
    g = GRAMS[ds_name]
    abl_all[ds_name] = run_ablation(DATASETS[ds_name]["X"], g["zz"], g["amp"], g["angle"],
                                     DATASETS[ds_name]["y"], ds_name)

fig, axes = plt.subplots(1, 3, figsize=(24, 7))
for ci, ds_name in enumerate(DATASETS):
    df_abl = pd.DataFrame(abl_all[ds_name]).T
    full_acc = df_abl.loc["Full Ensemble (4 models)", "acc"]
    rem = [c for c in df_abl.index if c.startswith("Remove")]
    deltas = [df_abl.loc[c, "acc"] - full_acc for c in rem]
    labels = [c.replace("Remove ", "-") for c in rem]
    colors = ["#FF6B6B" if d<0 else "#4ECDC4" for d in deltas]
    ax = axes[ci]
    ax.bar(range(len(labels)), deltas, color=colors, edgecolor="black", alpha=0.85)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.axhline(y=0, color="black", linewidth=1)
    ax.set_title(ds_name, fontsize=13, fontweight="bold")
    ax.set_ylabel("Delta Accuracy")
    for i, v in enumerate(deltas):
        ax.text(i, v+(0.001 if v>=0 else -0.003), f"{v:+.4f}", ha="center", fontsize=8, fontweight="bold")
plt.suptitle("Ablation: Accuracy Change When Removing Each Base Model",
             fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
save_fig(fig, "ablation_study.png")
plt.show()

for ds_name in DATASETS:
    df_abl = pd.DataFrame(abl_all[ds_name]).T
    print(f"\n=== {ds_name}: Ablation ===")
    print(df_abl.round(4).to_string())
    save_table(df_abl, f"ablation_{ds_name.lower().replace(' ','_').replace(chr(39),'')}")

# %% cell 16: Noise robustness + Clinical feature backtracking
print("="*60)
print("Noise Robustness + Clinical Feature Backtracking")
print("="*60)

def create_noisy_zz_kernel(nq, nl):
    dev = qml.device("default.mixed", wires=nq)
    @qml.qnode(dev, interface="autograd")
    def circuit(x1, x2):
        for i in range(nq):
            qml.Hadamard(wires=i); qml.RZ(x1[i], wires=i)
            if nl>0: qml.DepolarizingChannel(nl, wires=i)
        for i in range(nq-1):
            qml.IsingZZ(x1[i]*x1[i+1], wires=[i,i+1])
            if nl>0: qml.DepolarizingChannel(nl,wires=i); qml.DepolarizingChannel(nl,wires=i+1)
        for i in range(nq-2,-1,-1):
            qml.IsingZZ(-x2[i]*x2[i+1], wires=[i,i+1])
            if nl>0: qml.DepolarizingChannel(nl,wires=i); qml.DepolarizingChannel(nl,wires=i+1)
        for i in range(nq-1,-1,-1):
            qml.RZ(-x2[i], wires=i); qml.Hadamard(wires=i)
            if nl>0: qml.DepolarizingChannel(nl, wires=i)
        return qml.probs(wires=range(nq))
    return lambda x1, x2: circuit(x1, x2)[0]

def create_noisy_angle_kernel(nq, nl):
    dev = qml.device("default.mixed", wires=nq)
    @qml.qnode(dev, interface="autograd")
    def circuit(x1, x2):
        for i in range(nq):
            qml.RY(x1[i], wires=i)
            if nl>0: qml.DepolarizingChannel(nl, wires=i)
        for i in range(nq-1,-1,-1):
            qml.RY(-x2[i], wires=i)
            if nl>0: qml.DepolarizingChannel(nl, wires=i)
        return qml.probs(wires=range(nq))
    return lambda x1, x2: circuit(x1, x2)[0]

def create_noisy_amp_kernel(nq, nl):
    dev = qml.device("default.mixed", wires=nq)
    @qml.qnode(dev, interface="autograd")
    def circuit(x1, x2):
        qml.AmplitudeEmbedding(x1, wires=range(nq), normalize=True)
        if nl>0:
            for i in range(nq): qml.DepolarizingChannel(nl, wires=i)
        qml.adjoint(qml.AmplitudeEmbedding)(x2, wires=range(nq), normalize=True)
        if nl>0:
            for i in range(nq): qml.DepolarizingChannel(nl, wires=i)
        return qml.probs(wires=range(nq))
    return lambda x1, x2: circuit(x1, x2)[0]

def compute_noisy_gram(X, kfn, desc=""):
    n = len(X); gram = np.zeros((n,n))
    total = n*(n+1)//2
    with tqdm(total=total, desc=desc, leave=False) as pbar:
        for i in range(n):
            for j in range(i,n):
                v = kfn(X[i], X[j]); gram[i,j]=v; gram[j,i]=v; pbar.update(1)
    return np.clip((gram+gram.T)/2, 0, 1)

def eval_noise(X_sub, y_sub, nl):
    gz = compute_noisy_gram(X_sub, create_noisy_zz_kernel(8,nl), f"ZZ@{nl:.0%}")
    ga = compute_noisy_gram(X_sub, create_noisy_angle_kernel(8,nl), f"Ang@{nl:.0%}")
    gm = compute_noisy_gram(X_sub, create_noisy_amp_kernel(3,nl), f"Amp@{nl:.0%}")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    mnames = ["QSVM (ZZ Map)","QKNN (Amplitude)","QSVM (Angle)","Hard Voting Ensemble","OOF Stacking Ensemble"]
    all_m = {m:{"accuracy":[],"specificity":[]} for m in mnames}
    for tr, te in cv.split(X_sub, y_sub):
        y_tr, y_te = y_sub[tr], y_sub[te]
        p_zz = SVC(kernel="precomputed").fit(gz[tr][:,tr], y_tr).predict(gz[te][:,tr])
        p_ka = KNeighborsClassifier(n_neighbors=5, metric="precomputed").fit(
            np.clip(1-gm[tr][:,tr],0,None), y_tr).predict(np.clip(1-gm[te][:,tr],0,None))
        p_ang = SVC(kernel="precomputed").fit(ga[tr][:,tr], y_tr).predict(ga[te][:,tr])
        p_hv = mode(np.vstack((p_zz, p_ka, p_ang)), axis=0, keepdims=False).mode
        p_kn_ang = KNeighborsClassifier(n_neighbors=5, metric="precomputed").fit(
            np.clip(1-ga[tr][:,tr],0,None), y_tr).predict(np.clip(1-ga[te][:,tr],0,None))
        p_sa = SVC(kernel="precomputed").fit(gm[tr][:,tr], y_tr).predict(gm[te][:,tr])
        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        m_tr = np.zeros((len(tr),4))
        for ip, vp in inner.split(np.zeros(len(tr)), y_tr):
            it, iv = tr[ip], tr[vp]
            m_tr[vp,0] = SVC(kernel="precomputed").fit(gm[it][:,it], y_sub[it]).predict(gm[iv][:,it])
            m_tr[vp,1] = KNeighborsClassifier(n_neighbors=5, metric="precomputed").fit(
                np.clip(1-gm[it][:,it],0,None), y_sub[it]).predict(np.clip(1-gm[iv][:,it],0,None))
            m_tr[vp,2] = SVC(kernel="precomputed").fit(gz[it][:,it], y_sub[it]).predict(gz[iv][:,it])
            m_tr[vp,3] = KNeighborsClassifier(n_neighbors=5, metric="precomputed").fit(
                np.clip(1-ga[it][:,it],0,None), y_sub[it]).predict(np.clip(1-ga[iv][:,it],0,None))
        m_te = np.column_stack((p_sa, p_ka, p_zz, p_kn_ang))
        ml = LogisticRegression(random_state=42, class_weight="balanced").fit(m_tr, y_tr)
        p_st = ml.predict(m_te)
        for mn, pp in [("QSVM (ZZ Map)",p_zz),("QKNN (Amplitude)",p_ka),("QSVM (Angle)",p_ang),
                        ("Hard Voting Ensemble",p_hv),("OOF Stacking Ensemble",p_st)]:
            cm = compute_metrics(y_te, pp)
            all_m[mn]["accuracy"].append(cm["acc"]); all_m[mn]["specificity"].append(cm["spec"])
    return {m: {k: np.mean(v) for k,v in met.items()} for m, met in all_m.items()}

noise_results = {}
for ds_name in DATASETS:
    print(f"\n--- Noise: {ds_name} ---")
    X_full, y_full = DATASETS[ds_name]["X"], DATASETS[ds_name]["y"]
    if len(X_full) > NOISE_SUBSAMPLE:
        X_sub, _, y_sub, _ = train_test_split(X_full, y_full, train_size=NOISE_SUBSAMPLE,
                                               random_state=42, stratify=y_full)
    else:
        X_sub, y_sub = X_full, y_full
    noise_results[ds_name] = {}
    for nl in NOISE_LEVELS:
        print(f"  noise={nl:.0%}...")
        noise_results[ds_name][nl] = eval_noise(X_sub, y_sub, nl)

mnames_noise = ["QSVM (ZZ Map)","QKNN (Amplitude)","QSVM (Angle)","Hard Voting Ensemble","OOF Stacking Ensemble"]
fig, axes = plt.subplots(2, 3, figsize=(24, 12))
for ci, ds_name in enumerate(DATASETS):
    for ri, metric in enumerate(["accuracy","specificity"]):
        ax = axes[ri, ci]
        for mn in mnames_noise:
            vals = [noise_results[ds_name][nl][mn][metric] for nl in NOISE_LEVELS]
            lw = 2.5 if "Ensemble" in mn else 1.5
            ls = "-" if "Ensemble" in mn else "--"
            ax.plot([f"{nl:.0%}" for nl in NOISE_LEVELS], vals, "o-", label=mn, linewidth=lw, linestyle=ls)
        ax.set_title(f"{ds_name}" if ri==0 else "", fontsize=13, fontweight="bold")
        ax.set_ylabel(metric.capitalize()); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
plt.suptitle("Noise Robustness Under Depolarizing Noise", fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
save_fig(fig, "noise_robustness_actual_ensembles.png")
plt.show()

noise_deg_rows = []
for ds_name in DATASETS:
    clean = noise_results[ds_name][0.0]
    noisy = noise_results[ds_name][max(NOISE_LEVELS)]
    for mn in mnames_noise:
        noise_deg_rows.append({
            "Dataset": ds_name, "Model": mn,
            "Delta_Acc": round(clean[mn]["accuracy"]-noisy[mn]["accuracy"],3),
            "Delta_Spec": round(clean[mn]["specificity"]-noisy[mn]["specificity"],3)})
df_noise = pd.DataFrame(noise_deg_rows)
print("\n=== Noise Degradation ===")
print(df_noise.to_string(index=False))
save_table(df_noise, "noise_degradation")

print("\n--- Clinical Feature Backtracking ---")
def feature_backtrack(X_class, gram_zz, gram_amp, gram_angle, y, pca_obj, feat_names):
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    oof = {k: np.zeros(len(y)) for k in ["sa","ka","sz","kn"]}
    for tr, te in cv.split(X_class, y):
        y_tr = y[tr]
        oof["sa"][te] = SVC(kernel="precomputed").fit(gram_amp[tr][:,tr], y_tr).predict(gram_amp[te][:,tr])
        oof["ka"][te] = KNeighborsClassifier(n_neighbors=5, metric="precomputed").fit(
            np.clip(1-gram_amp[tr][:,tr],0,None), y_tr).predict(np.clip(1-gram_amp[te][:,tr],0,None))
        oof["sz"][te] = SVC(kernel="precomputed").fit(gram_zz[tr][:,tr], y_tr).predict(gram_zz[te][:,tr])
        oof["kn"][te] = KNeighborsClassifier(n_neighbors=5, metric="precomputed").fit(
            np.clip(1-gram_angle[tr][:,tr],0,None), y_tr).predict(np.clip(1-gram_angle[te][:,tr],0,None))
    X_meta = np.column_stack([oof[k] for k in oof])
    ml = LogisticRegression(random_state=42, class_weight="balanced").fit(X_meta, y)
    mc = np.abs(ml.coef_[0])
    loadings = pca_obj.components_
    ev = pca_obj.explained_variance_ratio_
    comp_imp = ev * np.sum(mc)
    raw_imp = np.zeros(len(feat_names))
    for c in range(len(comp_imp)):
        raw_imp += np.abs(loadings[c, :]) * comp_imp[c]
    return (raw_imp / raw_imp.sum()) * 100

bt_results = {}
for ds_name in DATASETS:
    g = GRAMS[ds_name]
    ds = DATASETS[ds_name]
    bt_results[ds_name] = feature_backtrack(
        ds["X"], g["zz"], g["amp"], g["angle"], ds["y"], ds["pca"], ds["raw_names"])

fig, axes = plt.subplots(1, 3, figsize=(28, 8))
for ci, ds_name in enumerate(DATASETS):
    raw_imp = bt_results[ds_name]
    feat_names = DATASETS[ds_name]["raw_names"]
    si = np.argsort(raw_imp)[::-1]
    n_show = min(15, len(feat_names))
    top = si[:n_show]
    ax = axes[ci]
    colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, n_show))
    ax.barh(range(n_show), raw_imp[top][::-1], color=colors[::-1], edgecolor="black", alpha=0.85)
    ax.set_yticks(range(n_show)); ax.set_yticklabels([feat_names[i] for i in top][::-1], fontsize=9)
    ax.set_xlabel("Importance (%)"); ax.set_title(ds_name, fontsize=14, fontweight="bold")
    for i, v in enumerate(raw_imp[top][::-1]):
        ax.text(v+0.3, i, f"{v:.1f}%", va="center", fontsize=8, fontweight="bold")
plt.suptitle("Clinical Feature Backtracking", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
save_fig(fig, "clinical_feature_backtracking.png")
plt.show()

for ds_name in DATASETS:
    df_bt = pd.DataFrame({"Feature": DATASETS[ds_name]["raw_names"],
                           "Importance (%)": bt_results[ds_name]}).sort_values(
        "Importance (%)", ascending=False).reset_index(drop=True)
    df_bt.index += 1; df_bt.index.name = "Rank"
    print(f"\n=== {ds_name}: Top Features ===")
    print(df_bt.head(10).to_string())
    save_table(df_bt, f"backtrack_{ds_name.lower().replace(' ','_').replace(chr(39),'')}")

# %% cell 17: Master bar chart (auto-generated from parametric results)
print("="*60)
print("Master Bar Chart — All Models Across All Datasets")
print("="*60)

models_ordered = [
    "Classical RBF", "QSVM (Angle)", "QSVM (ZZ Map)", "QSVM (Amplitude)",
    "QKNN (Angle)", "QKNN (ZZ Map)", "QKNN (Amplitude)", "Pure QNN (VQC)",
    "Hard Voting Ensemble", "OOF Stacking Ensemble",
]
model_labels = [m.replace(" ", "\n").replace("(", "\n(") if len(m)>12 else m for m in models_ordered]

metrics_list = ["acc", "prec", "rec", "spec"]
metric_labels = ["Accuracy", "Precision", "Recall", "Specificity"]
metric_colors = ["#3498DB", "#2ECC71", "#F39C12", "#E74C3C"]

fig, axes = plt.subplots(1, 3, figsize=(26, 9))
x = np.arange(len(models_ordered))
width = 0.2

for di, ds_name in enumerate(DATASETS):
    folds = PARAM_FOLDS[ds_name]
    ax = axes[di]
    for mi, (mk, ml_label) in enumerate(zip(metrics_list, metric_labels)):
        vals = [np.mean(folds[m][mk]) for m in models_ordered]
        offset = (mi - 1.5) * width
        ax.bar(x + offset, vals, width, label=ml_label if di == 0 else "",
               color=metric_colors[mi], edgecolor="white", linewidth=0.5, alpha=0.85)
    ax.set_title(ds_name, fontsize=15, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(" ","\n") for m in models_ordered], fontsize=7.5, ha="right", rotation=35)
    ax.set_ylim(0, 1.12); ax.grid(axis="y", alpha=0.2)
    ax.set_ylabel("Score" if di == 0 else "")
    ax.axvspan(8.5, 9.5, alpha=0.08, color="red")
    ax.axvspan(6.5, 7.5, alpha=0.08, color="gray")

handles = [plt.Rectangle((0,0),1,1, facecolor=c, alpha=0.85) for c in metric_colors]
fig.legend(handles, metric_labels, loc="lower center", ncol=4, fontsize=12,
           bbox_to_anchor=(0.5, -0.01), frameon=True)
plt.suptitle("Master Performance Comparison: All Models Across All Datasets",
             fontsize=17, fontweight="bold", y=1.01)
plt.tight_layout(rect=[0, 0.04, 1, 0.98])
save_fig(fig, "master_bar_chart.png")
plt.show()

print("\n" + "="*60)
print(f"ALL DONE. Outputs in: {FIG_DIR}, {TBL_DIR}, {CKP_DIR}")
print(f"PILOT_MODE was: {PILOT_MODE}")
if PILOT_MODE:
    print("Set PILOT_MODE = False and rerun for full results.")
print("="*60)