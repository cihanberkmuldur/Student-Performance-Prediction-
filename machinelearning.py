import pandas as pd
import numpy as np
import math
import sys
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, log_loss
)

# Veri Yedekleme


def load_data(filepath):
    df = pd.read_csv(filepath, sep=';')
    if 'G3' not in df.columns:
        df = pd.read_csv(filepath, sep=',')
    df.columns = df.columns.str.strip()
    return df

df_mat = load_data("student-mat.csv")
df_por = load_data("student-por.csv")


# Regresyon Metrikleri


def regression_metrics(y_true, y_pred, X_train):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    n = len(y_true)
    k = X_train.shape[1] + 1

    if mse > 0:
        aic = n * math.log(mse) + 2 * k
        bic = n * math.log(mse) + k * math.log(n)
    else:
        aic = np.nan
        bic = np.nan

    hist, _ = np.histogram(y_pred, bins=10)
    prob = hist / np.sum(hist)
    entropy_val = -np.sum(prob * np.log(prob + 1e-9))

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "AIC": aic,
        "BIC": bic,
        "Entropy": entropy_val
    }


# Regresyon Analizi Seneryo 2


def run_regression(df, ders_adi, senaryo):
    print(f"\n===== REGRESYON ANALÄ°ZÄ° ({ders_adi}) | {senaryo} =====")

    y = df["G3"]
    X_raw = df.drop(columns=["G3"])

    if senaryo == "Sadece Sosyal":
        X_raw = X_raw.drop(columns=["G1", "G2"], errors="ignore")

    X = pd.get_dummies(X_raw, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        train_metrics = regression_metrics(y_train, train_pred, X_train)
        test_metrics = regression_metrics(y_test, test_pred, X_train)

        print(f"\n{name}")
        print("EÄŸitim:", train_metrics)
        print("Test   :", test_metrics)


# Sınıflandırma Metrikleri


def classification_metrics(y_true, y_pred, y_proba):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    sens = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    ce = log_loss(y_true, y_proba)

    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    gmean = np.sqrt(sens * spec)

    return {
        "ACC": acc,
        "Precision": prec,
        "Sensitivity": sens,
        "F1": f1,
        "CrossEntropy": ce,
        "G-Mean": gmean
    }


# Sınıflandırma Analizi Seneryo 2

def run_classification(df, ders_adi, senaryo):
    print(f"\n===== SINIFLANDIRMA ANALÄ°ZÄ° ({ders_adi}) | {senaryo} =====")

    df = df.copy()
    df["G3_cls"] = (df["G3"] >= 10).astype(int)

    y = df["G3_cls"]
    X_raw = df.drop(columns=["G3", "G3_cls"])

    if senaryo == "Sadece Sosyal":
        X_raw = X_raw.drop(columns=["G1", "G2"], errors="ignore")

    X = pd.get_dummies(X_raw, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        train_proba = model.predict_proba(X_train)
        test_proba = model.predict_proba(X_test)

        print(f"\n{name}")
        print("EÄŸitim:", classification_metrics(y_train, train_pred, train_proba))
        print("Test   :", classification_metrics(y_test, test_pred, test_proba))
def confusion_matrix_output(df, ders_adi, senaryo):
    print(f"\n===== CONFUSION MATRIX ({ders_adi}) | {senaryo} =====")

    df = df.copy()
    df["G3_cls"] = (df["G3"] >= 10).astype(int)

    y = df["G3_cls"]
    X_raw = df.drop(columns=["G3", "G3_cls"])

    if senaryo == "Sadece Sosyal":
        X_raw = X_raw.drop(columns=["G1", "G2"], errors="ignore")

    X = pd.get_dummies(X_raw, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel()

    print("Confusion Matrix:")
    print(cm)
    print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
def correlation_matrix(df, ders_adi):
    print(f"\n===== KORELASYON MATRÄ°SÄ° ({ders_adi}) =====")

    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    print(corr["G3"].sort_values(ascending=False))
def correlation_heatmap(df, ders_adi):
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    plt.figure(figsize=(10, 8))
    plt.imshow(corr, cmap="coolwarm")
    plt.colorbar()
    plt.xticks(range(len(corr)), corr.columns, rotation=90)
    plt.yticks(range(len(corr)), corr.columns)
    plt.title(f"Korelasyon Matrisi - {ders_adi}")
    plt.tight_layout()
    plt.show()



# Tahmin


if __name__ == "__main__":

    for senaryo in ["Notlar Dahil", "Sadece Sosyal"]:
        run_regression(df_mat, "Matematik", senaryo)
        run_regression(df_por, "Portekizce", senaryo)

        run_classification(df_mat, "Matematik", senaryo)
        run_classification(df_por, "Portekizce", senaryo)
        confusion_matrix_output(df_mat, "Matematik", senaryo)
        confusion_matrix_output(df_por, "Portekizce", senaryo)
        correlation_matrix(df_mat, "Matematik")
        correlation_matrix(df_por, "Portekizce")
        correlation_heatmap(df_mat, "Matematik")
        correlation_heatmap(df_por, "Portekizce")



        
