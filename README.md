# BNPL Financial Default Risk — Multi-Class Classification

Machine Learning (AIML) · Assignment 2 · Five classifiers compared on a
Buy-Now-Pay-Later default-risk dataset, served through an interactive Streamlit app.

---

## a. Problem statement

Buy-Now-Pay-Later (BNPL) lenders approve small, unsecured instalment credit at
checkout in seconds, with no traditional underwriting interview. The lender
therefore has to decide, from a thin profile of behavioural and bureau data,
how likely a customer is to miss their instalments.

**Task.** Given a customer's demographic profile, income, bureau credit score and
current BNPL borrowing behaviour, predict their **default risk** as one of three
ordered classes — **Low**, **Medium** or **High**.

This is a **multi-class (3-class) classification** problem. Two properties shape
every modelling decision in this repo:

1. **The classes are heavily imbalanced** — 88.0% of customers are Low risk,
   6.8% Medium and 5.2% High. A model that predicts "Low" for everybody already
   scores 88% accuracy, so **accuracy alone is a misleading metric here**. MCC and
   macro-averaged F1 are used as the primary yardsticks instead, because both
   collapse when the minority classes are ignored.
2. **The error costs are asymmetric.** Approving a customer who defaults (a High
   case predicted Low) costs the lender the principal; declining a good customer
   costs a margin. Recall on the High class therefore matters more than raw
   accuracy, which is why every model that supports it is trained with
   `class_weight="balanced"`.

---

## b. Dataset description

| Property | Value |
|---|---|
| Name | BNPL Financial Default Risk Dataset |
| Instances | **10,000** customers (requirement: ≥ 500) |
| Raw columns | 11 (1 ID + 9 predictors + 1 target) |
| **Features fed to the models** | **18** after feature engineering (requirement: ≥ 12) |
| Target | `Default_Risk` — `Low` / `Medium` / `High` |
| Task type | Multi-class classification |
| Missing values | `Income_USD` 305 (3.05%), `Credit_Score` 298 (2.98%) |
| Duplicate rows | 0 |

### Raw columns

| Column | Type | Description | Range / levels |
|---|---|---|---|
| `Customer_ID` | ID | Unique customer key — **dropped** before modelling | C00001–C10000 |
| `Age` | numeric | Customer age in years | 18 – 64 |
| `Employment_Status` | categorical | Primary income source | Employed (60.3%), Student (19.7%), Freelancer (15.0%), Unemployed (5.1%) |
| `Income_USD` | numeric | Declared annual income | 5,000 – 139,452 |
| `Credit_Score` | numeric | Bureau credit score | 300 – 850 |
| `Total_BNPL_Active_Loans` | numeric | Open BNPL instalment plans | 0 – 10 |
| `Total_BNPL_Debt_USD` | numeric | Outstanding BNPL balance | 0 – 2,731 |
| `Late_Payment_History` | categorical | Any prior late instalment | No (75.2%), Yes (24.8%) |
| `Shopping_Category_Most_Frequent` | categorical | Dominant spend category | Fashion, Electronics, Home/Furniture, Travel, Groceries/Essentials |
| `Average_Transaction_Value_USD` | numeric | Mean basket size | 10 – 1,420 |
| `Default_Risk` | **target** | Risk band | Low 8,801 · Medium 678 · High 521 |

### Target distribution

| Class | Rows | Share |
|---|---|---|
| Low | 8,801 | 88.01% |
| Medium | 678 | 6.78% |
| High | 521 | 5.21% |

### Feature engineering (9 raw predictors → 18 model features)

The raw file carries only 9 predictive columns, below the 12-feature minimum, and
the raw columns are also weak on their own: the mean age, income and basket size
are nearly identical across the three risk bands. The signal lives in the
**ratios between** the columns, not in the columns themselves. Nine derived
features are therefore added in `features.py`:

| Derived feature | Definition | Why it was added |
|---|---|---|
| `Monthly_Income_USD` | `Income_USD / 12` | Puts income on the same cadence as an instalment plan |
| `Debt_To_Monthly_Income` | `Total_BNPL_Debt_USD / Monthly_Income` | **The strongest single signal.** Class means: Low 0.08, Medium 0.37, High 0.96 |
| `Debt_Per_Active_Loan` | `Total_BNPL_Debt_USD / max(loans, 1)` | Separates "many small plans" from "one large plan" |
| `Txn_To_Monthly_Income` | `Avg_Transaction_Value / Monthly_Income` | Flags customers whose typical basket is large relative to earnings |
| `Credit_Score_Per_Loan` | `Credit_Score / max(loans, 1)` | Spreads creditworthiness across parallel commitments |
| `Income_Was_Missing` | missingness indicator | Lets a model react to an absent value instead of silently trusting the imputed median |
| `Credit_Score_Was_Missing` | missingness indicator | Same, for the bureau score |
| `Age_Band` | 18-24 / 25-34 / 35-44 / 45-54 / 55+ | Gives the linear models a non-linear read on age |
| `Credit_Score_Band` | Poor / Fair / Good / Very Good / Excellent / Unknown | Standard bureau bands; `Unknown` is reserved for a missing score |

Every derived column is a function of that row alone, so deriving them before the
train/test split leaks nothing. Both missingness flags are computed **before** the
imputer runs, so the information that a value was absent survives imputation.

> **Note on the missingness flags.** They were added as good practice, but the
> missingness in this dataset turns out to be uninformative: the class
> distribution for rows with a missing income (4.9% / 88.5% / 6.6%) is
> statistically indistinguishable from rows without (5.2% / 88.0% / 6.8%). They
> are retained because they cost nothing and make the imputation honest.

Final feature count: **13 numeric + 5 categorical = 18 features**, which expand to
35 columns after one-hot encoding.

### Preprocessing

All preprocessing lives **inside** each saved scikit-learn `Pipeline`, so a saved
model is self-contained and no transformation can leak between the split halves:

- **Numeric** → median imputation → `StandardScaler`
- **Categorical** → most-frequent imputation → `OneHotEncoder(handle_unknown="ignore")`

Scaling is essential for Logistic Regression, KNN and the RBF SVM, and harmless for
the Decision Tree and Gaussian Naive Bayes, so one shared preprocessor keeps all
five models comparable on identical inputs. `handle_unknown="ignore"` means a
category never seen in training will not crash the deployed app.

**Split:** stratified 80 / 20 → 8,000 train / 2,000 test, `random_state=42`. The
hold-out half is written to `test_data.csv` and is the file the Streamlit app
scores. No model ever sees it during training or tuning.

---

## c. GitHub repository link

**https://github.com/Epsom700/bnpl-default-risk-ml**

**Live Streamlit app:** **https://bits-2026aj05312-bnpl.streamlit.app/**

### Repository structure

```
bnpl-default-risk-ml/
├── app.py                  # Streamlit front-end (scoring only, never retrains)
├── features.py             # Shared schema + feature engineering (imported by both)
├── train_models.py         # Trains, tunes and persists all five pipelines
├── requirements.txt        # Pinned dependencies
├── README.md
├── test_data.csv           # 2,000-row stratified hold-out split
├── data/
│   └── BNPL_Financial_Default_Risk_Dataset.csv
├── notebooks/
│   └── BNPL_Model_Development.ipynb   # EDA + full modelling walkthrough
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── svm.joblib
    ├── metrics.json        # Hold-out metrics + confusion matrices
    └── metadata.json       # Split, feature and hyper-parameter bookkeeping
```

`features.py` is imported by **both** `train_models.py` and `app.py`. That is what
guarantees the columns reaching a pipeline at scoring time are byte-for-byte the
columns it was fitted on, and it keeps the `.joblib` files free of custom classes
that could fail to unpickle on the deployment host.

### Reproducing the results

```bash
pip install -r requirements.txt
python train_models.py      # ≈ 8 seconds; rewrites model/ and test_data.csv
streamlit run app.py
```

---

## d. Models used

All five classifiers are trained on the **same** 8,000-row training split with the
**same** 18 features and the **same** preprocessor. Each one is tuned with
`GridSearchCV` optimising **macro F1** (5-fold; 3-fold for the SVM), and then
scored once on the untouched 2,000-row hold-out split.

| Model | Tuned over | Selected |
|---|---|---|
| Logistic Regression | `C` ∈ {0.1, 1, 10} | `C=10`, `class_weight=balanced` |
| Decision Tree | `max_depth` ∈ {4, 6, 8, None}, `min_samples_leaf` ∈ {1, 5, 20} | `max_depth=None`, `min_samples_leaf=5`, `class_weight=balanced` |
| KNN | `n_neighbors` ∈ {5, 11, 21}, `weights` ∈ {uniform, distance} | `n_neighbors=5`, `weights=distance` |
| Naive Bayes | `var_smoothing` ∈ {1e-9, 1e-6, 1e-3} | `var_smoothing=1e-3` (Gaussian) |
| Support Vector Machine | `C` ∈ {1, 10}, `gamma` ∈ {scale, 0.1} | `C=10`, `gamma=scale`, RBF, `class_weight=balanced` |

### Comparison table — 2,000-row hold-out split

Precision, Recall and F1 are **macro-averaged**; AUC is **one-vs-rest macro**. Macro
averaging is used deliberately: it weights the 104-row High class equally with the
1,760-row Low class, so a model cannot hide a failure on the minority classes.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9450 | 0.9905 | 0.8202 | 0.9332 | 0.8639 | 0.7951 |
| Decision Tree | 0.9490 | 0.9405 | 0.8005 | 0.8706 | 0.8324 | 0.7865 |
| KNN | 0.9445 | 0.9562 | 0.8233 | 0.7558 | 0.7859 | 0.7288 |
| Naive Bayes | 0.8205 | 0.9346 | 0.5190 | 0.6923 | 0.5624 | 0.4737 |
| **Support Vector Machine** | **0.9605** | **0.9912** | **0.8546** | 0.9149 | **0.8805** | **0.8353** |

Best value per column in bold. Logistic Regression takes the highest macro recall
(0.9332); the SVM wins the other five metrics.

### Per-class recall — where the models actually differ

Overall accuracy spans a narrow 0.82–0.96, which hides the real story. Recall on
the two minority classes is what separates these models:

| Model | Recall (Low) | Recall (Medium) | Recall (High) |
|---|---|---|---|
| Logistic Regression | 0.948 | 0.890 | **0.962** |
| Decision Tree | 0.968 | 0.750 | 0.894 |
| KNN | 0.989 | 0.500 | 0.779 |
| Naive Bayes | 0.854 | 0.338 | 0.885 |
| Support Vector Machine | 0.971 | 0.860 | 0.913 |

KNN illustrates the trap precisely: its 0.9445 accuracy is within half a point of
the SVM's, yet it misses **half** of all Medium-risk customers.

---

## Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | Far stronger than a linear model has any right to be here (AUC 0.9905, MCC 0.7951 — second overall), and it posts the single best macro recall of all five, 0.9332, catching 96.2% of High-risk customers. The reason is the feature engineering, not the model: `Debt_To_Monthly_Income` turns a ratio relationship the raw columns could not express into a near-linear separator, and the log-odds boundary then fits it almost perfectly. Its weakness is precision on Medium (0.563) — it over-flags 90 Low customers as Medium, trading false alarms for safety. It is also the only model whose decision is directly auditable from its coefficients, which matters in a regulated credit setting. **Best choice when recall and explainability outrank precision.** |
| **Decision Tree** | Edges out Logistic Regression on raw accuracy (0.9490 vs 0.9450) but is clearly weaker where it counts: MCC 0.7865 and the **worst AUC of all five models, 0.9405**. The gap between its good accuracy and poor AUC is the signature of a model producing hard, poorly calibrated probabilities — an unpruned tree's leaves return near-0/1 confidences, so ranking customers by risk score barely works even when the top-1 label is right. Grid search chose `min_samples_leaf=5` with unlimited depth, which still overfits: 22 of 136 Medium customers are pushed all the way to High. Useful for the readable if-then rules it yields, not for scoring. |
| **KNN** | **The weakest of the four competent models (MCC 0.7288, macro F1 0.7859)**, and the clearest cautionary tale about judging on accuracy. At 0.9445 accuracy it looks competitive with the SVM, yet it recovers only **50% of Medium-risk customers** and 77.9% of High. Two causes: it has no `class_weight` parameter, so the 88% Low majority dominates every neighbourhood vote, and with 35 one-hot dimensions the distance metric is diluted by binary columns that contribute little real geometry. `weights="distance"` and `k=5` were selected — the smallest k in the grid — confirming that larger neighbourhoods only drown the minority classes further. |
| **Naive Bayes** | **Comfortably the worst model on every metric** (accuracy 0.8205, macro F1 0.5624, MCC 0.4737) and the only one that scores below the 88% majority-class baseline on accuracy. Its core assumption — features conditionally independent given the class — is violated twice over by this feature set: the engineered ratios are by construction correlated with the columns they are built from (`Debt_To_Monthly_Income` with both `Total_BNPL_Debt_USD` and `Income_USD`), and one-hot indicator columns are mutually exclusive, not independent, while also being badly non-Gaussian. The result is wildly overconfident posteriors: precision on High collapses to 0.371 and on Medium to 0.194, as it mislabels 179 Low customers as Medium and another 78 as High. Note that its AUC is still a respectable 0.9346 — it **ranks** customers acceptably but places its decision thresholds in the wrong places. |
| **Support Vector Machine** | **The winner — best on five of the six metrics**: accuracy 0.9605, AUC 0.9912, macro precision 0.8546, macro F1 0.8805, MCC 0.8353. The RBF kernel with `C=10` carves a curved boundary that handles the one genuinely hard region in this dataset, the overlap where Medium (`Debt_To_Monthly_Income` 0.26–0.47) meets Low and High, and it is the only model that stays strong on **both** minority classes at once (recall 0.860 Medium / 0.913 High) rather than trading one against the other. Its confusion matrix is the cleanest of the five: **not one** of the 104 High-risk customers is classified as Low — the single most expensive error type in BNPL lending — and the 9 that it does miss are downgraded only as far as Medium, where they still get flagged for review. Costs: the slowest to train (2.5 s vs 0.2–1.7 s), `probability=True` requires internal cross-validation, and it offers no native explanation of an individual decision. |
| **Overall, Winner for your dataset?** | **Support Vector Machine (RBF kernel).** It leads on accuracy, AUC, macro precision, macro F1 and MCC, and its 0.8353 MCC is a decisive 0.04 clear of the runner-up on a metric that is genuinely hard to inflate under 88/7/5 class imbalance. It also avoids the business-critical error entirely: **0** of the 104 High-risk customers are classified as Low, against 3 for the Decision Tree and 5 for KNN (Logistic Regression and Naive Bayes also reach 0 here). **Logistic Regression is the recommended fallback:** it is within 0.04 MCC of the SVM, actually beats it on macro recall (0.9332 vs 0.9149) and on High-risk recall (0.962 vs 0.913), trains in a fraction of the time, and produces coefficients a credit-risk committee can inspect — which in a regulated lending context may outweigh the SVM's margin. |

### Why MCC and not accuracy

With an 88% majority class, a constant "Low" predictor scores **0.8801 accuracy** —
which would rank it second of five on accuracy alone. Its MCC would be **0.0**.
That gap is the whole argument for the metric choice in this report: MCC uses all
four quadrants of the confusion matrix and only rises when a model genuinely
discriminates. Naive Bayes is the live demonstration — at 0.8205 accuracy it is
*worse than predicting "Low" for everybody*, yet it is unambiguously doing real
(if poorly calibrated) work, which only MCC and macro F1 reveal.

---

## Streamlit app

**Live app:** **https://bits-2026aj05312-bnpl.streamlit.app/**

### Required features

| Requirement | Where it is implemented |
|---|---|
| **Dataset upload option (CSV)** | Sidebar → *1. Test data* → file uploader. A bundled-`test_data.csv` checkbox gives a working demo if nothing is uploaded. Uploads are validated against the required schema, with a clear error naming any missing column. |
| **Model selection dropdown** | Sidebar → *2. Model* → dropdown over all five classifiers, plus an *Also score all 5 models* toggle. |
| **Display of evaluation metrics** | *Selected model* tab shows all six metrics; *All 5 models* tab shows the full comparison table with the best value per column highlighted, a grouped bar chart, and a CSV download. |
| **Confusion matrix / classification report** | *Selected model* tab renders a seaborn heatmap next to the per-class classification report; *All 5 models* tab renders all five matrices side by side. |

Plus a **Predictions** tab with row-level actual vs predicted labels, per-class
probabilities, a misclassified-rows-only filter, and a CSV export.

### No retraining on upload

The app calls `joblib.load()` on the five fitted pipelines and only ever calls
`.predict()` / `.predict_proba()`. There is no `.fit()` call anywhere in `app.py`.
Scoring the bundled `test_data.csv` through the app reproduces the comparison
table above **exactly**, to all four decimal places, which is the check that the
saved models and the reported metrics are the same artefacts.

### Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Deploying on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to <https://streamlit.io/cloud> and sign in with GitHub.
3. **New app** → select the repository → branch `main` → main file `app.py`.
4. **Deploy**.

The `model/` directory is committed, so the deployed app loads the fitted
pipelines directly and never trains in the cloud. Dependency versions are pinned
in `requirements.txt` because scikit-learn will refuse to unpickle a model saved
by a different minor version.

---

## Notes on academic integrity

- The dataset was processed, split and modelled from scratch in `train_models.py`;
  no pre-built notebook or template was copied.
- The nine engineered features in `features.py` are specific to BNPL credit risk
  and were derived from the EDA in `notebooks/BNPL_Model_Development.ipynb`.
- All reported metrics are reproducible: run `python train_models.py` and compare
  against `model/metrics.json`. `random_state=42` is fixed throughout.
