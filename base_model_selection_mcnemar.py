import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from statsmodels.stats.contingency_tables import mcnemar as mcnemar_test_fn

CHECKPOINT_DIR = "checkpoints"

PARK_PATH = "/kaggle/input/datasets/miitdaga/uci-parkinsons-disease-dataset/parkinsons.data"
DIAB_PATH = "/kaggle/input/datasets/organizations/uciml/pima-indians-diabetes-database/diabetes.csv"

y_park = pd.read_csv(PARK_PATH)["status"].values
y_bc = load_breast_cancer().target
y_diab = pd.read_csv(DIAB_PATH)["Outcome"].values

datasets = {
    "Parkinson's": {
        "y": y_park,
        "angle": np.load(f"{CHECKPOINT_DIR}/gram_angle_parkinsons.npy"),
        "zz": np.load(f"{CHECKPOINT_DIR}/gram_zz_parkinsons.npy"),
    },
    "Breast Cancer": {
        "y": y_bc,
        "angle": np.load(f"{CHECKPOINT_DIR}/gram_angle_breast_cancer.npy"),
        "zz": np.load(f"{CHECKPOINT_DIR}/gram_zz_breast_cancer.npy"),
    },
    "Diabetes": {
        "y": y_diab,
        "angle": np.load(f"{CHECKPOINT_DIR}/gram_angle_diabetes.npy"),
        "zz": np.load(f"{CHECKPOINT_DIR}/gram_zz_diabetes.npy"),
    },
}

cv = StratifiedKFold(10, shuffle=True, random_state=42)

pairs = [
    ("QSVM (Angle)", "QKNN (Angle)", "angle"),
    ("QSVM (ZZ Map)", "QKNN (ZZ Map)", "zz"),
]

for ds_name, ds in datasets.items():
    print(f"=== {ds_name} ===")
    y = ds["y"]

    for name_a, name_b, gram_key in pairs:
        g = ds[gram_key]
        p_svm = np.zeros(len(y))
        p_knn = np.zeros(len(y))

        for tr, te in cv.split(g, y):
            p_svm[te] = SVC(kernel="precomputed").fit(g[tr][:,tr], y[tr]).predict(g[te][:,tr])
            p_knn[te] = KNeighborsClassifier(5, metric="precomputed").fit(
                np.clip(1-g[tr][:,tr], 0, None), y[tr]).predict(np.clip(1-g[te][:,tr], 0, None))

        agree = int(np.sum(p_svm == p_knn))
        ca = (p_svm == y).astype(int)
        cb = (p_knn == y).astype(int)
        n01 = int(np.sum((ca==1) & (cb==0)))
        n10 = int(np.sum((ca==0) & (cb==1)))
        table = np.array([[int(np.sum((ca==1) & (cb==1))), n01],
                          [n10, int(np.sum((ca==0) & (cb==0)))]])
        res = mcnemar_test_fn(table, exact=(n01+n10 < 25), correction=(n01+n10 >= 25))

        print(f"  {name_a} vs {name_b}:")
        print(f"    Acc: {accuracy_score(y, p_svm):.4f} vs {accuracy_score(y, p_knn):.4f}")
        print(f"    Agreement: {agree}/{len(y)} ({100*agree/len(y):.1f}%)")
        print(f"    Discordant: n01={n01}, n10={n10}")
        print(f"    McNemar p={res.pvalue:.4f}")
        print()
