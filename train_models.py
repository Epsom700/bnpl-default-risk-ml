"""Train and persist the five classifiers for BNPL default-risk prediction.

Run from the repository root:

    python train_models.py

Outputs
-------
model/<slug>.joblib   one self-contained Pipeline per classifier
model/metrics.json    hold-out metrics + confusion matrices for the app
model/metadata.json   dataset / split / feature bookkeeping
test_data.csv         the 20% hold-out split in the RAW schema, for the app
"""

import json
import os
import time
import warnings
from pathlib import Path

# numpy 2.x on macOS/Accelerate raises spurious overflow / divide-by-zero
# RuntimeWarnings from inside the lbfgs gradient matmul. The fitted coefficients
# stay finite and the scores are unaffected, so only that specific noise is
# filtered. PYTHONWARNINGS is set as well so the joblib worker processes that
# GridSearchCV spawns inherit the filter - they do not inherit warnings.filters.
# PYTHONWARNINGS matches its message field literally from the start of the
# string, so a regex cannot be used there; the filter is scoped by category and
# by the two sklearn modules that emit the noise instead. ConvergenceWarning is
# a UserWarning, so genuine fit problems are still reported.
_WORKER_FILTERS = ",".join(
    f"ignore::RuntimeWarning:{m}"
    for m in ("sklearn.utils.extmath", "sklearn.linear_model._linear_loss")
)
os.environ["PYTHONWARNINGS"] = (
    f"{os.environ['PYTHONWARNINGS']},{_WORKER_FILTERS}"
    if os.environ.get("PYTHONWARNINGS")
    else _WORKER_FILTERS
)
warnings.filterwarnings(
    "ignore", message=".*encountered in matmul", category=RuntimeWarning
)

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

import features as F

RANDOM_STATE = 42
TEST_SIZE = 0.20
DATA_PATH = Path("data/BNPL_Financial_Default_Risk_Dataset.csv")
MODEL_DIR = Path("model")
TEST_CSV = Path("test_data.csv")


def build_preprocessor() -> ColumnTransformer:
    """Median-impute + scale the numerics, mode-impute + one-hot the categoricals.

    Scaling matters for Logistic Regression, KNN and the RBF SVM; it is
    harmless for the Decision Tree and Gaussian Naive Bayes, so a single shared
    preprocessor keeps the five models comparable on identical inputs.
    """
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, F.NUMERIC_FEATURES),
            ("cat", categorical, F.CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


# Each entry: slug -> (display name, estimator, hyper-parameter grid, cv folds).
# `class_weight="balanced"` is used wherever the estimator supports it, because
# 88% of the rows are Low risk and the business cost sits on the High class.
MODEL_SPECS = {
    "logistic_regression": (
        "Logistic Regression",
        LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        {"clf__C": [0.1, 1.0, 10.0]},
        5,
    ),
    "decision_tree": (
        "Decision Tree",
        DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
        {
            "clf__max_depth": [4, 6, 8, None],
            "clf__min_samples_leaf": [1, 5, 20],
        },
        5,
    ),
    "knn": (
        "KNN",
        KNeighborsClassifier(),
        {
            "clf__n_neighbors": [5, 11, 21],
            "clf__weights": ["uniform", "distance"],
        },
        5,
    ),
    "naive_bayes": (
        "Naive Bayes",
        GaussianNB(),
        {"clf__var_smoothing": [1e-9, 1e-6, 1e-3]},
        5,
    ),
    "svm": (
        "Support Vector Machine",
        SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        {"clf__C": [1.0, 10.0], "clf__gamma": ["scale", 0.1]},
        3,
    ),
}


def evaluate(name, pipe, X_test, y_test, classes):
    """Hold-out metrics for one fitted pipeline."""
    y_pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)
    # roc_auc_score requires `labels` in sorted order, while `classes` is kept in
    # risk order (Low -> High) for readable confusion matrices. Score the AUC
    # against the sorted labels with the probability columns realigned to match.
    auc_labels = sorted(classes)
    order = [list(pipe.classes_).index(c) for c in auc_labels]
    auc_proba = proba[:, order]

    return {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "auc_ovr_macro": roc_auc_score(
            y_test, auc_proba, multi_class="ovr", average="macro", labels=auc_labels
        ),
        "precision_macro": precision_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_test, y_pred),
        "precision_weighted": precision_score(
            y_test, y_pred, average="weighted", zero_division=0
        ),
        "recall_weighted": recall_score(
            y_test, y_pred, average="weighted", zero_division=0
        ),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=classes).tolist(),
        "per_class": classification_report(
            y_test, y_pred, labels=classes, zero_division=0, output_dict=True
        ),
    }


def main():
    MODEL_DIR.mkdir(exist_ok=True)

    raw = pd.read_csv(DATA_PATH)
    missing = F.check_schema(raw)
    if missing:
        raise SystemExit(f"dataset is missing required columns: {missing}")
    print(f"loaded {DATA_PATH}: {raw.shape[0]} rows x {raw.shape[1]} columns")

    classes = [c for c in F.CLASS_ORDER if c in set(raw[F.TARGET])]
    y = raw[F.TARGET]

    # Split the RAW rows so test_data.csv keeps the original schema; feature
    # derivation is row-wise, so deriving before or after the split is identical.
    raw_train, raw_test = train_test_split(
        raw, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    raw_test.to_csv(TEST_CSV, index=False)
    print(f"wrote {TEST_CSV}: {len(raw_test)} hold-out rows")

    X_train = F.build_feature_frame(raw_train)
    X_test = F.build_feature_frame(raw_test)
    y_train, y_test = raw_train[F.TARGET], raw_test[F.TARGET]
    print(f"features used: {X_train.shape[1]} ({len(F.NUMERIC_FEATURES)} numeric, "
          f"{len(F.CATEGORICAL_FEATURES)} categorical)")

    results, best_params = [], {}
    for slug, (name, estimator, grid, folds) in MODEL_SPECS.items():
        started = time.time()
        search = GridSearchCV(
            Pipeline([("prep", build_preprocessor()), ("clf", estimator)]),
            param_grid=grid,
            scoring="f1_macro",
            cv=folds,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)
        pipe = search.best_estimator_

        joblib.dump(pipe, MODEL_DIR / f"{slug}.joblib")
        best_params[slug] = {k: str(v) for k, v in search.best_params_.items()}

        row = evaluate(name, pipe, X_test, y_test, classes)
        row["slug"] = slug
        row["cv_f1_macro"] = search.best_score_
        row["best_params"] = best_params[slug]
        row["fit_seconds"] = round(time.time() - started, 1)
        results.append(row)

        print(
            f"  {name:<24} acc={row['accuracy']:.4f} auc={row['auc_ovr_macro']:.4f} "
            f"f1m={row['f1_macro']:.4f} mcc={row['mcc']:.4f} "
            f"({row['fit_seconds']}s, {search.best_params_})"
        )

    results.sort(key=lambda r: r["mcc"], reverse=True)
    (MODEL_DIR / "metrics.json").write_text(
        json.dumps(
            {
                "classes": classes,
                "winner": results[0]["model"],
                "results": results,
            },
            indent=2,
        )
    )
    (MODEL_DIR / "metadata.json").write_text(
        json.dumps(
            {
                "dataset": DATA_PATH.name,
                "n_rows": int(raw.shape[0]),
                "n_raw_columns": int(raw.shape[1]),
                "n_features_used": int(X_train.shape[1]),
                "numeric_features": F.NUMERIC_FEATURES,
                "categorical_features": F.CATEGORICAL_FEATURES,
                "target": F.TARGET,
                "classes": classes,
                "class_counts": raw[F.TARGET].value_counts().to_dict(),
                "test_size": TEST_SIZE,
                "random_state": RANDOM_STATE,
                "n_train": int(len(raw_train)),
                "n_test": int(len(raw_test)),
                "best_params": best_params,
            },
            indent=2,
        )
    )
    print(f"\nwinner by MCC: {results[0]['model']}")
    print(f"saved 5 pipelines + metrics.json + metadata.json to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
