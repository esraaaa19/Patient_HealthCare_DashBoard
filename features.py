"""Shared cleaning + feature lists. Mirrors the notebook exactly so the
dashboard model matches the one reported in the paper."""
import os
import numpy as np
import pandas as pd

NUMERIC = [
    "admission_type_id", "admission_source_id", "time_in_hospital",
    "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient",
    "number_diagnoses",
]

MEDICATIONS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone",
]

CATEGORICAL = (
    ["race", "gender", "age", "payer_code", "medical_specialty"]
    + MEDICATIONS
    + ["change", "diabetesMed", "diag_1_group", "diag_2_group", "diag_3_group"]
)

FEATURES = NUMERIC + CATEGORICAL   # 10 numeric + 33 categorical = 43


def find_data_file(name):
    """Look in data/ first, then the repo root (works locally and on GitHub)."""
    for p in (os.path.join("data", name), name):
        if os.path.exists(p):
            return p
    return os.path.join("data", name)


def classify_diagnosis(code):
    """ICD-9 grouping, copied from the notebook."""
    if pd.isna(code):
        return "Unknown"
    code = str(code)
    if code.startswith("V"):
        return "Supplementary"
    if code.startswith("E"):
        return "External Cause"
    try:
        num = float(code)
    except ValueError:
        return "Other"
    if 250 <= num < 251:
        return "Diabetes"
    elif 390 <= num < 460:
        return "Circulatory"
    elif 460 <= num < 520:
        return "Respiratory"
    elif 520 <= num < 580:
        return "Digestive"
    elif 580 <= num < 630:
        return "Genitourinary"
    elif 680 <= num < 710:
        return "Skin"
    elif 710 <= num < 740:
        return "Musculoskeletal"
    elif 780 <= num < 800:
        return "Symptoms/Signs"
    elif 800 <= num < 1000:
        return "Injury/Complications"
    else:
        return "Other"


def load_clean(path=None):
    """Same cleaning steps as the notebook. Keeps all columns (for EDA);
    the model only uses FEATURES."""
    path = path or find_data_file("diabetic_data.csv")
    df = pd.read_csv(path).replace("?", np.nan)
    df["readmitted_30"] = np.where(df["readmitted"] == "<30", 1, 0)

    a1c = df["A1Cresult"]                      # dropped from the model, kept for the KPI
    df["hba1c_tested"] = a1c.notna() & (a1c != "None")

    df["race"] = df["race"].fillna("Unknown")
    df["medical_specialty"] = df["medical_specialty"].fillna("Unknown")
    df["payer_code"] = df["payer_code"].fillna("Unknown")

    for col in ["diag_1", "diag_2", "diag_3"]:
        df[col + "_group"] = df[col].apply(classify_diagnosis)
    return df


def load_health_clean(path=None):
    """Patient healthcare dataset, cleaned exactly like the notebook:
    drop duplicates, parse dates, length of stay, absolute billing,
    admission year/month/day and age groups."""
    path = path or find_data_file("healthcare_dataset.csv")
    h = pd.read_csv(path).drop_duplicates()
    h["Date of Admission"] = pd.to_datetime(h["Date of Admission"])
    h["Discharge Date"] = pd.to_datetime(h["Discharge Date"])
    h["Length of Stay"] = (h["Discharge Date"] - h["Date of Admission"]).dt.days
    h["Billing Amount"] = h["Billing Amount"].abs()
    h["Admission Year"] = h["Date of Admission"].dt.year
    h["Admission Month"] = h["Date of Admission"].dt.month
    h["Admission Day"] = h["Date of Admission"].dt.day_name()
    h["Age Group"] = pd.cut(
        h["Age"], bins=[0, 18, 30, 45, 60, 75, 100],
        labels=["Under 18", "18-29", "30-44", "45-59", "60-74", "75+"],
        right=False)
    return h.reset_index(drop=True)
