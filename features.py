"""Shared schema and feature engineering, used by train_models.py and app.py."""

import numpy as np
import pandas as pd

ID_COL = "Customer_ID"
TARGET = "Default_Risk"
CLASS_ORDER = ["Low", "Medium", "High"]

RAW_NUMERIC = [
    "Age",
    "Income_USD",
    "Credit_Score",
    "Total_BNPL_Active_Loans",
    "Total_BNPL_Debt_USD",
    "Average_Transaction_Value_USD",
]
RAW_CATEGORICAL = [
    "Employment_Status",
    "Late_Payment_History",
    "Shopping_Category_Most_Frequent",
]
REQUIRED_COLUMNS = RAW_NUMERIC + RAW_CATEGORICAL + [TARGET]

DERIVED_NUMERIC = [
    "Monthly_Income_USD",
    "Debt_To_Monthly_Income",
    "Debt_Per_Active_Loan",
    "Txn_To_Monthly_Income",
    "Credit_Score_Per_Loan",
    "Income_Was_Missing",
    "Credit_Score_Was_Missing",
]
DERIVED_CATEGORICAL = ["Age_Band", "Credit_Score_Band"]

NUMERIC_FEATURES = RAW_NUMERIC + DERIVED_NUMERIC
CATEGORICAL_FEATURES = RAW_CATEGORICAL + DERIVED_CATEGORICAL
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

AGE_BANDS = [0, 24, 34, 44, 54, np.inf]
AGE_LABELS = ["18-24", "25-34", "35-44", "45-54", "55+"]
CREDIT_BANDS = [0, 580, 670, 740, 800, np.inf]
CREDIT_LABELS = ["Poor", "Fair", "Good", "Very Good", "Excellent"]

_EPS = 1.0  # guards the ratio denominators against zero income


def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features. Row-wise only, so no leakage across the split."""
    out = df.copy()

    out["Income_Was_Missing"] = out["Income_USD"].isna().astype(int)
    out["Credit_Score_Was_Missing"] = out["Credit_Score"].isna().astype(int)

    monthly_income = out["Income_USD"] / 12.0
    out["Monthly_Income_USD"] = monthly_income

    # strongest single signal in the data
    out["Debt_To_Monthly_Income"] = out["Total_BNPL_Debt_USD"] / (monthly_income + _EPS)

    out["Debt_Per_Active_Loan"] = out["Total_BNPL_Debt_USD"] / out[
        "Total_BNPL_Active_Loans"
    ].clip(lower=1)

    out["Txn_To_Monthly_Income"] = out["Average_Transaction_Value_USD"] / (
        monthly_income + _EPS
    )

    out["Credit_Score_Per_Loan"] = out["Credit_Score"] / out[
        "Total_BNPL_Active_Loans"
    ].clip(lower=1)

    out["Age_Band"] = pd.cut(
        out["Age"], bins=AGE_BANDS, labels=AGE_LABELS, right=True
    ).astype(object)
    out["Credit_Score_Band"] = pd.cut(
        out["Credit_Score"], bins=CREDIT_BANDS, labels=CREDIT_LABELS, right=False
    ).astype(object)
    out["Credit_Score_Band"] = out["Credit_Score_Band"].where(
        out["Credit_Score"].notna(), "Unknown"
    )

    return out


def check_schema(df: pd.DataFrame) -> list:
    """Return the list of `REQUIRED_COLUMNS` absent from `df`."""
    return [c for c in REQUIRED_COLUMNS if c not in df.columns]


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Derive features and return only the columns the pipelines expect."""
    return derive_features(df)[FEATURE_COLUMNS]
