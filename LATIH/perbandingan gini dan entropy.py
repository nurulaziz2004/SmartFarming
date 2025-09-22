import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

# ===================== Load Dataset =====================
df = pd.read_csv("D:/skripsi/Kontrol Kipas/selada sudah ada kipas.csv")

FEATURES = ["suhu", "kelembaban", "kelembaban_tanah", "intensitas_cahaya"]

# ===================== Fungsi Evaluasi =====================
def evaluate_and_compare(X, y, name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    results = {}

    for criterion in ["gini", "entropy"]:
        # Train model
        model = DecisionTreeClassifier(criterion=criterion, max_depth=4, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Akurasi
        acc_train = accuracy_score(y_train, model.predict(X_train))
        acc_test = accuracy_score(y_test, y_pred)
        cv_score = cross_val_score(model, X, y, cv=5).mean()

        # Simpan hasil
        results[criterion] = {
            "train_acc": acc_train,
            "test_acc": acc_test,
            "cv_acc": cv_score
        }

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix - {name} ({criterion.capitalize()})")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

        # ROC Curve
        if len(np.unique(y_test)) == 2:
            y_proba = model.predict_proba(X_test)[:,1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            plt.figure()
            plt.plot(fpr, tpr, label=f"{criterion.capitalize()} AUC={roc_auc:.3f}")
            plt.plot([0,1],[0,1],"r--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve - {name} ({criterion})")
            plt.legend()
            plt.show()

        # Feature Importance
        fi = model.feature_importances_
        plt.figure(figsize=(6,4))
        sns.barplot(x=fi, y=FEATURES, orient="h")
        plt.title(f"Feature Importance - {name} ({criterion})")
        plt.show()

        # Rules
        print(f"\n=== Rules {name} ({criterion}) ===")
        print(export_text(model, feature_names=FEATURES, decimals=2))

    # Plot perbandingan akurasi
    metrics = ["train_acc", "test_acc", "cv_acc"]
    df_acc = pd.DataFrame(results).T[metrics]

    plt.figure(figsize=(7,5))
    df_acc.plot(kind="bar")
    plt.title(f"Perbandingan Gini vs Entropy - {name}")
    plt.ylabel("Accuracy")
    plt.ylim(0.9, 1.0)
    plt.xticks(rotation=0)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

    return results

# ===================== Run untuk Siram dan Kipas =====================
X = df[FEATURES]
print("=== Model SIRAM ===")
results_siram = evaluate_and_compare(X, df["label"], "Siram")

print("\n=== Model KIPAS ===")
results_kipas = evaluate_and_compare(X, df["kipas_exhaust"], "Kipas")
