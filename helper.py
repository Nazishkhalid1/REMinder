import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

le_dict = {}
scaler = None
label_map = {0: "None", 1: "Insomnia", 2: "Sleep Apnea"}
categorical = ['Gender', 'Occupation', 'BMI Category', 'Blood Pressure', 'Smoking', 'Alcohol Consumption']
numeric_cols = ['Age', 'Sleep Duration', 'Physical Activity Level', 'Stress Level']
drop_cols = ['Person ID', 'Quality of Sleep', 'Caffeine Consumption']

def train_model():
    global le_dict, scaler
    df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")
    df.columns = df.columns.str.strip()
    df = df.dropna()

    if "Smoking" not in df.columns:
        df["Smoking"] = "No"
    if "Alcohol Consumption" not in df.columns:
        df["Alcohol Consumption"] = "No"
    if "Caffeine Consumption" not in df.columns:
        df["Caffeine Consumption"] = 100

    df["Sleep Disorder"] = df["Sleep Disorder"].str.strip().replace({"Normal": "None"})
    allowed_labels = ["None", "Insomnia", "Sleep Apnea"]
    df = df[df["Sleep Disorder"].isin(allowed_labels)]

    print("Unique labels in cleaned 'Sleep Disorder':", df["Sleep Disorder"].unique())

    for col in categorical:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le

    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    df = df.drop(columns=drop_cols, errors='ignore')

    X = df.drop('Sleep Disorder', axis=1)
    y = df['Sleep Disorder']
    print("Training target labels:", y.unique())
    print("Features used for training:", list(X.columns))

    model = RandomForestClassifier()
    feature_order = X.columns.tolist()
    model.feature_order = feature_order
    model.fit(X, y)

    return model, preprocess_input

def safe_transform(le, series, col_name):
    known = set(le.classes_)
    unknowns = [val for val in series.unique() if val not in known]
    if unknowns:
        print(f"[WARNING] Column '{col_name}' has unknown labels: {unknowns}")
        series = series.apply(lambda x: x if x in known else le.classes_[0])
    return le.transform(series)

def preprocess_input(input_df):
    global le_dict, scaler
    df = input_df.copy()
    df.columns = df.columns.str.strip()
    df = df.drop(columns=drop_cols, errors='ignore')
    for col, le in le_dict.items():
        df[col] = safe_transform(le, df[col].astype(str), col)
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    return df

def predict_sleep_disorder(model, preprocess_fn, input_df):
    feature_order = model.feature_order
    df = preprocess_fn(input_df)
    df = df[feature_order]
    pred = model.predict(df)[0]
    advice = get_sleep_advice(pred)
    return pred, advice

def get_sleep_advice(label):
    advice = {
        "Insomnia": (
            "Try setting a consistent bedtime, limit screen time an hour before sleep, "
            "avoid caffeine late in the day, and consider relaxation techniques like deep breathing or meditation."
        ),
        "Sleep Apnea": (
            "Consider seeking a sleep study or consulting a doctor. Weight loss, avoiding alcohol, "
            "and sleeping on your side can help. CPAP therapy is a common treatment."
        ),
        "None": (
            "Great job! Maintain your sleep habits by keeping a regular schedule, staying active during the day, "
            "and managing stress. Continue monitoring your sleep quality."
        )
    }
    return advice.get(label, "No specific advice available.")