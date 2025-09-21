import os, time, threading
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, validation_curve
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

# ===================== Load Dataset =====================
df = pd.read_csv("D:/skripsi/Kontrol Kipas/selada sudah ada kipas.csv")

FEATURES = ["suhu", "kelembaban", "kelembaban_tanah", "intensitas_cahaya"]
TARGET_SIRAM = "label"
TARGET_KIPAS = "kipas_exhaust"

X = df[FEATURES]
y_siram = df[TARGET_SIRAM]
y_kipas = df[TARGET_KIPAS]

# ===================== Split Data =====================
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X, y_siram, test_size=0.30, random_state=42)
X_train_k, X_test_k, y_train_k, y_test_k = train_test_split(X, y_kipas, test_size=0.30, random_state=42)

# ===================== Train Models =====================
model_siram = DecisionTreeClassifier(max_depth=4, random_state=42)
model_kipas = DecisionTreeClassifier(max_depth=4, random_state=42)
model_siram.fit(X_train_s, y_train_s)
model_kipas.fit(X_train_k, y_train_k)

# ===================== Evaluation =====================
def evaluate_model(model, X_train, y_train, X_test, y_test, name):
    y_pred = model.predict(X_test)
    acc_train = accuracy_score(y_train, model.predict(X_train))
    acc_test = accuracy_score(y_test, y_pred)

    print(f"\n===== {name} =====")
    print("Train Accuracy:", acc_train)
    print("Test Accuracy :", acc_test)
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Class 0","Class 1"],
                yticklabels=["Class 0","Class 1"])
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"static/cm_{name}.png", dpi=140)
    plt.close()

    # ROC Curve
    if len(np.unique(y_test)) == 2:
        y_proba = model.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        plt.plot([0,1],[0,1],'r--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {name}")
        plt.legend()
        plt.savefig(f"static/roc_{name}.png", dpi=140)
        plt.close()

# Evaluasi model
evaluate_model(model_siram, X_train_s, y_train_s, X_test_s, y_test_s, "Siram")
evaluate_model(model_kipas, X_train_k, y_train_k, X_test_k, y_test_k, "Kipas")

# ===================== Feature Importance =====================
def plot_feature_importance(model, features, name):
    importance = model.feature_importances_
    plt.figure(figsize=(6,4))
    sns.barplot(x=importance, y=features, orient="h", color="skyblue")
    plt.title(f"Feature Importance - {name}")
    plt.tight_layout()
    plt.savefig(f"static/featimp_{name}.png", dpi=140)
    plt.close()
    return importance

imp_siram = plot_feature_importance(model_siram, FEATURES, "Siram")
imp_kipas = plot_feature_importance(model_kipas, FEATURES, "Kipas")

# ===================== Cross Validation =====================
cv_siram = cross_val_score(model_siram, X, y_siram, cv=5).mean()
cv_kipas = cross_val_score(model_kipas, X, y_kipas, cv=5).mean()
print("\nCross-validation (Siram):", cv_siram)
print("Cross-validation (Kipas):", cv_kipas)

# ===================== Learning Curve =====================
def plot_learning_curve(model, X, y, name):
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5)
    plt.figure()
    plt.plot(train_sizes, train_scores.mean(axis=1), "o-", label="Train")
    plt.plot(train_sizes, test_scores.mean(axis=1), "o-", label="Test")
    plt.xlabel("Training examples")
    plt.ylabel("Accuracy")
    plt.title(f"Learning Curve - {name}")
    plt.legend()
    plt.savefig(f"static/learning_{name}.png", dpi=140)
    plt.close()

plot_learning_curve(model_siram, X, y_siram, "Siram")
plot_learning_curve(model_kipas, X, y_kipas, "Kipas")

# ===================== Rules Export =====================
rules_siram = export_text(model_siram, feature_names=FEATURES)
rules_kipas = export_text(model_kipas, feature_names=FEATURES)

with open("static/rules_siram.txt", "w") as f: f.write(rules_siram)
with open("static/rules_kipas.txt", "w") as f: f.write(rules_kipas)
