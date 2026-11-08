"""Streamlit app: scores an uploaded test CSV with the saved models (no retraining)."""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
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

import features as F

MODEL_DIR = Path("model")
DEFAULT_TEST_CSV = Path("test_data.csv")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "auc_ovr_macro": "AUC (OvR macro)",
    "precision_macro": "Precision (macro)",
    "recall_macro": "Recall (macro)",
    "f1_macro": "F1 (macro)",
    "mcc": "MCC",
}

st.set_page_config(
    page_title="BNPL Default Risk Classifier",
    page_icon="💳",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading trained pipelines...")
def load_models():
    loaded, missing = {}, []
    for name, fname in MODEL_FILES.items():
        path = MODEL_DIR / fname
        if path.exists():
            loaded[name] = joblib.load(path)
        else:
            missing.append(str(path))
    return loaded, missing


@st.cache_data(show_spinner=False)
def load_json(path: str):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_data(show_spinner=False)
def read_csv(file_bytes: bytes) -> pd.DataFrame:
    from io import BytesIO

    return pd.read_csv(BytesIO(file_bytes))


def score(pipe, X, y_true, classes):
    """Metrics for one fitted pipeline on the uploaded rows."""
    y_pred = pipe.predict(X)
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }

    present = sorted(set(y_true) & set(pipe.classes_))
    if len(present) >= 2 and hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X)[
            :, [list(pipe.classes_).index(c) for c in present]
        ]
        if len(present) == 2:
            out["auc_ovr_macro"] = roc_auc_score(y_true, proba[:, 1], labels=present)
        else:
            if len(present) < len(pipe.classes_):
                # some class missing from the file, renormalise
                proba = proba / proba.sum(axis=1, keepdims=True)
            out["auc_ovr_macro"] = roc_auc_score(
                y_true, proba, multi_class="ovr", average="macro", labels=present
            )
    else:
        out["auc_ovr_macro"] = float("nan")

    labels = [c for c in classes if c in set(y_true) | set(y_pred)]
    out["_y_pred"] = y_pred
    out["_labels"] = labels
    out["_cm"] = confusion_matrix(y_true, y_pred, labels=labels)
    out["_report"] = classification_report(
        y_true, y_pred, labels=labels, zero_division=0, output_dict=True
    )
    return out


def plot_confusion(cm, labels, title):
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- sidebar ----
models, missing = load_models()
meta = load_json("model/metadata.json")
train_metrics = load_json("model/metrics.json")

st.sidebar.title("💳 BNPL Default Risk")
st.sidebar.caption(
    "Five scikit-learn classifiers predicting a customer's Buy-Now-Pay-Later "
    "default risk as Low, Medium or High."
)

if missing:
    st.error(
        "These model files are missing - run `python train_models.py` first:\n\n"
        + "\n".join(f"- `{m}`" for m in missing)
    )
    st.stop()

st.sidebar.subheader("1. Test data")
uploaded = st.sidebar.file_uploader(
    "Upload test CSV",
    type="csv",
    help="Must contain the 9 raw input columns plus the Default_Risk target.",
)
use_bundled = st.sidebar.checkbox(
    f"Use bundled {DEFAULT_TEST_CSV.name}",
    value=True,
    disabled=uploaded is not None,
    help="The 20% stratified hold-out split that the models never saw in training.",
)

st.sidebar.subheader("2. Model")
model_choice = st.sidebar.selectbox(
    "Select a model to inspect",
    list(MODEL_FILES.keys()),
    index=list(MODEL_FILES.keys()).index("Support Vector Machine"),
)
show_all = st.sidebar.checkbox("Also score all 5 models", value=True)

if meta:
    st.sidebar.divider()
    st.sidebar.markdown(
        f"**Trained on** {meta['n_train']:,} rows  \n"
        f"**Hold-out** {meta['n_test']:,} rows  \n"
        f"**Features** {meta['n_features_used']}  \n"
        f"**Classes** {', '.join(meta['classes'])}"
    )
    with st.sidebar.expander("Chosen hyper-parameters"):
        for slug, params in meta["best_params"].items():
            st.markdown(f"**{slug}**")
            st.json(params, expanded=False)

# -------------------------------------------------------------- load rows ----
st.title("BNPL Default Risk - Model Comparison")

if uploaded is not None:
    df = read_csv(uploaded.getvalue())
    source = f"uploaded file `{uploaded.name}`"
elif use_bundled and DEFAULT_TEST_CSV.exists():
    df = pd.read_csv(DEFAULT_TEST_CSV)
    source = f"bundled `{DEFAULT_TEST_CSV.name}`"
else:
    st.info(
        "Upload a test CSV from the sidebar, or tick **Use bundled test_data.csv** "
        "to score the stratified hold-out split shipped with the repo."
    )
    st.markdown(
        "**Required columns**\n\n"
        + "\n".join(f"- `{c}`" for c in F.REQUIRED_COLUMNS)
    )
    st.stop()

absent = F.check_schema(df)
if absent:
    st.error(
        "The CSV is missing these required columns: "
        + ", ".join(f"`{c}`" for c in absent)
    )
    st.stop()

df = df.dropna(subset=[F.TARGET])
unknown = sorted(set(df[F.TARGET]) - set(F.CLASS_ORDER))
if unknown:
    st.warning(
        f"Dropping {len(df[df[F.TARGET].isin(unknown)]):,} rows whose "
        f"{F.TARGET} is not one of {F.CLASS_ORDER}: {unknown}"
    )
    df = df[~df[F.TARGET].isin(unknown)]
if df.empty:
    st.error("No scorable rows left after validation.")
    st.stop()

X = F.build_feature_frame(df)
y = df[F.TARGET]
classes = [c for c in F.CLASS_ORDER if c in set(y)]

st.caption(
    f"Scoring **{len(df):,} rows** from {source} with pre-trained pipelines "
    "(no retraining)."
)

c1, c2, c3 = st.columns(3)
c1.metric("Rows scored", f"{len(df):,}")
c2.metric("Input features derived", X.shape[1])
c3.metric("Classes present", len(classes))

with st.expander("Preview the data and the class balance"):
    st.dataframe(df.head(15), use_container_width=True)
    counts = y.value_counts().reindex(classes).rename("Rows").to_frame()
    counts["Share"] = (counts["Rows"] / len(df)).map("{:.1%}".format)
    st.dataframe(counts, use_container_width=False)

# ------------------------------------------------------------ scoring tabs ---
scored = {model_choice: score(models[model_choice], X, y, classes)}
if show_all:
    for name, pipe in models.items():
        if name not in scored:
            scored[name] = score(pipe, X, y, classes)

tab_single, tab_compare, tab_predict = st.tabs(
    ["Selected model", "All 5 models", "Predictions"]
)

with tab_single:
    res = scored[model_choice]
    st.subheader(f"{model_choice} - evaluation metrics")
    items = list(METRIC_LABELS.items())
    for row_start in (0, 3):
        for col, (key, label) in zip(st.columns(3), items[row_start : row_start + 3]):
            value = res[key]
            col.metric(label, "n/a" if np.isnan(value) else f"{value:.4f}")

    left, right = st.columns([1, 1.2])
    with left:
        st.markdown("**Confusion matrix**")
        st.pyplot(plot_confusion(res["_cm"], res["_labels"], model_choice))
    with right:
        st.markdown("**Classification report**")
        report = pd.DataFrame(res["_report"]).T
        # the scalar "accuracy" entry gets broadcast into every column, support included
        if "accuracy" in report.index:
            report.loc["accuracy", "support"] = len(y)
        report["support"] = report["support"].astype(int)
        st.dataframe(
            report.style.format(
                {"precision": "{:.4f}", "recall": "{:.4f}", "f1-score": "{:.4f}"}
            ),
            use_container_width=True,
        )
        if train_metrics:
            saved = next(
                (r for r in train_metrics["results"] if r["model"] == model_choice),
                None,
            )
            if saved:
                st.caption(
                    "Reference scores on the original hold-out split: "
                    f"accuracy {saved['accuracy']:.4f}, "
                    f"AUC {saved['auc_ovr_macro']:.4f}, "
                    f"F1 {saved['f1_macro']:.4f}, MCC {saved['mcc']:.4f}."
                )

with tab_compare:
    if not show_all:
        st.info("Tick **Also score all 5 models** in the sidebar to fill this table.")
    else:
        st.subheader("Comparison on the current test data")
        table = pd.DataFrame(
            [
                {"ML Model Name": name, **{METRIC_LABELS[k]: r[k] for k in METRIC_LABELS}}
                for name, r in scored.items()
            ]
        ).set_index("ML Model Name")
        table = table.reindex([n for n in MODEL_FILES if n in table.index])
        st.dataframe(
            table.style.format("{:.4f}").highlight_max(axis=0, color="#a5d6a7"),
            use_container_width=True,
        )

        best = table["MCC"].idxmax()
        st.success(
            f"Best model on this data by MCC: **{best}** "
            f"(MCC {table.loc[best, 'MCC']:.4f}, "
            f"accuracy {table.loc[best, 'Accuracy']:.4f})"
        )

        st.markdown("**Metric comparison**")
        fig, ax = plt.subplots(figsize=(9, 3.6))
        table.plot(kind="bar", ax=ax, width=0.8, colormap="viridis")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=12)
        ax.legend(fontsize=7, ncol=3, loc="lower right")
        fig.tight_layout()
        st.pyplot(fig)

        st.markdown("**Confusion matrix per model**")
        ordered = [n for n in MODEL_FILES if n in scored]
        for row_start in range(0, len(ordered), 3):
            chunk = ordered[row_start : row_start + 3]
            for col, name in zip(st.columns(3), chunk):
                with col:
                    r = scored[name]
                    st.pyplot(plot_confusion(r["_cm"], r["_labels"], name))

        st.download_button(
            "Download comparison table (CSV)",
            table.to_csv().encode(),
            file_name="model_comparison.csv",
            mime="text/csv",
        )

with tab_predict:
    res = scored[model_choice]
    st.subheader(f"Row-level predictions - {model_choice}")
    pred_df = df[[F.ID_COL]].copy() if F.ID_COL in df.columns else pd.DataFrame(index=df.index)
    pred_df["Actual"] = y.values
    pred_df["Predicted"] = res["_y_pred"]
    pred_df["Correct"] = pred_df["Actual"] == pred_df["Predicted"]

    pipe = models[model_choice]
    proba = pipe.predict_proba(X)
    for cls in [c for c in F.CLASS_ORDER if c in pipe.classes_]:
        pred_df[f"P({cls})"] = proba[:, list(pipe.classes_).index(cls)].round(4)

    only_wrong = st.checkbox("Show only misclassified rows", value=False)
    view = pred_df[~pred_df["Correct"]] if only_wrong else pred_df
    st.caption(f"{len(view):,} of {len(pred_df):,} rows shown.")
    st.dataframe(view, use_container_width=True, height=420)
    st.download_button(
        "Download predictions (CSV)",
        pred_df.to_csv(index=False).encode(),
        file_name=f"predictions_{model_choice.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )
