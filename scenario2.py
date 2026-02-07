# SUDHIKSHA 24BAD117
# SCENARIO 2 – MULTI-FILE VERSION

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score
)

files = [
    "LICI - minute ata.csv",
    "LICI - 3 minute data.csv",
    "LICI - 5 minute data.csv",
    "LICI - 10 minute data.csv",
    "LICI - Daily data.csv"
]


def find_col(df, possible):
    for name in possible:
        if name in df.columns:
            return name
    return None


for file in files:

    print("\n" + "="*50)
    print("Processing file:", file)

    df = pd.read_csv(file)
    df.columns = df.columns.str.strip().str.lower()

    open_col   = find_col(df, ['open'])
    high_col   = find_col(df, ['high'])
    low_col    = find_col(df, ['low'])
    close_col  = find_col(df, ['close', 'adj close', 'adj_close'])
    volume_col = find_col(df, ['volume'])

    if None in [open_col, high_col, low_col, close_col, volume_col]:
        print("Required columns missing → Skipping file")
        continue

    df['price_movement'] = np.where(df[close_col] > df[open_col], 1, 0)
    df = df[[open_col, high_col, low_col, volume_col, 'price_movement']].dropna()

    X = df[[open_col, high_col, low_col, volume_col]]
    y = df['price_movement']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    cm   = confusion_matrix(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_prob)

    print("\n--- Performance Metrics ---")
    print("Accuracy :", acc)
    print("Precision:", prec)
    print("Recall   :", rec)
    print("F1-Score :", f1)
    print("ROC-AUC  :", auc)
    print("Confusion Matrix:\n", cm)


    # ROC Curve 
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve – {file}")
    plt.grid()
    plt.show()

    # Feature Importance 
    importance = pd.Series(model.coef_[0], index=X.columns)

    plt.figure()
    importance.plot(kind="bar")
    plt.title(f"Feature Importance – {file}")
    plt.ylabel("Coefficient Value")
    plt.grid()
    plt.show()

  
    # Hyperparameter tuning
    param_grid = {
        "C": [0.01, 0.1, 1, 10],
        "penalty": ["l2"],
        "solver": ["lbfgs"]
    }

    grid = GridSearchCV(
        LogisticRegression(max_iter=1000),
        param_grid,
        cv=5,
        scoring="accuracy"
    )

    grid.fit(X_train, y_train)

    print("\nBest Hyperparameters:", grid.best_params_)
    print("Tuned Accuracy:", accuracy_score(y_test, grid.best_estimator_.predict(X_test)))

print("\nAll files processed successfully.")
