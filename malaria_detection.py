# ============================================================
#         MALARIA DETECTION WITH MACHINE LEARNING
# ============================================================
#
# HOW TO RUN:
#   1. Install libraries:
#      pip install numpy pandas scikit-learn matplotlib seaborn
#
#   2. Put dataset.csv in the same folder as this file
#
#   3. Run:
#      python malaria_detection.py
#
# DATASET INFO:
#   - File      : dataset.csv
#   - Label col : Label  →  "Parasitized" or "Uninfected"
#   - Features  : area_0, area_1, area_2, area_3, area_4
#   - Samples   : 27,558
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# STEP 1 — LOAD DATASET
# ============================================================

def load_data():
    print("\n" + "=" * 50)
    print("  📂 DATASET SELECTION")
    print("=" * 50)
    print("  [1] Use default dataset  (dataset.csv)")
    print("  [2] Load my own dataset  (enter filename)")
    print("-" * 50)

    while True:
        choice = input("  Enter 1 or 2: ").strip()
        if choice in ["1", "2"]:
            break
        print("  ⚠️  Please enter 1 or 2.")

    if choice == "1":
        filename = "dataset.csv"
    else:
        filename = input("  Enter your filename (e.g. mydata.csv): ").strip()
        if not filename.endswith(".csv"):
            filename += ".csv"

    print(f"\n📂 Loading: {filename}")

    try:
        df = pd.read_csv(filename)
        print(f"   ✅ Loaded! Shape: {df.shape}")
        print(f"   ✅ Columns: {list(df.columns)}")
    except FileNotFoundError:
        print(f"   ❌ '{filename}' not found. Make sure it is in the same folder.")
        exit()

    # --- Auto-detect label column ---
    possible_label_names = ["Label", "label", "target", "Target",
                            "class", "Class", "diagnosis", "Diagnosis",
                            "result", "Result", "output", "Output"]

    label_col = None
    for name in possible_label_names:
        if name in df.columns:
            label_col = name
            break

    if label_col is None:
        print("\n   ⚠️  Could not find a label column automatically.")
        print("   Your columns are:")
        for i, col in enumerate(df.columns):
            print(f"     [{i}] {col}")
        idx       = int(input("\n   Enter the NUMBER of your label column: ").strip())
        label_col = df.columns[idx]

    if label_col != "Label":
        print(f"   ℹ️  Renaming column '{label_col}' → 'Label'")
        df = df.rename(columns={label_col: "Label"})

    print(f"   ✅ Label column : '{label_col}'")
    print(f"   ✅ Labels       : {df['Label'].value_counts().to_dict()}")

    return df


# ============================================================
# STEP 2 — EXPLORE THE DATA
# ============================================================

def explore_data(df):
    print("\n🔍 Dataset Overview")
    print("-" * 40)
    print(df.head())
    print("\nShape         :", df.shape)
    print("\nMissing values:\n", df.isnull().sum())
    print("\nBasic statistics:")
    print(df.describe().round(2))


# ============================================================
# STEP 3 — VISUALIZE THE DATA
# ============================================================

def visualize_data(df):
    print("\n📈 Creating visualizations...")

    features = ["area_0", "area_1", "area_2", "area_3", "area_4"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Malaria Detection — Feature Analysis", fontsize=16, fontweight="bold")
    axes_flat = axes.flatten()

    colors = ["#2ecc71", "#e74c3c"]   # green = Uninfected, red = Parasitized

    for i, feature in enumerate(features):
        ax = axes_flat[i]
        for label_name, color in zip(["Uninfected", "Parasitized"], colors):
            data = df[df["Label"] == label_name][feature]
            ax.hist(data, bins=30, alpha=0.6, color=color, label=label_name)
        ax.set_title(feature.replace("_", " ").title())
        ax.set_xlabel("Area Value")
        ax.set_ylabel("Count")
        ax.legend()

    # Hide the unused 6th subplot
    axes_flat[5].set_visible(False)

    plt.tight_layout()
    plt.show()


# ============================================================
# STEP 4 — PREPARE DATA FOR TRAINING
# ============================================================

def prepare_data(df):
    print("\n⚙️  Preparing data for training...")

    # Convert text labels → numbers
    # "Parasitized" → 1,  "Uninfected" → 0
    le = LabelEncoder()
    df["label_encoded"] = le.fit_transform(df["Label"])
    print(f"   ✅ Label encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    X = df[["area_0", "area_1", "area_2", "area_3", "area_4"]]
    y = df["label_encoded"]

    # 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print(f"   ✅ Training samples : {len(X_train)}")
    print(f"   ✅ Testing samples  : {len(X_test)}")

    return X_train, X_test, y_train, y_test, scaler, le


# ============================================================
# STEP 5 — TRAIN THE MODEL
# ============================================================

def train_model(X_train, y_train):
    print("\n🤖 Training the ML model (this may take a few seconds)...")

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    print("   ✅ Training complete!")
    return model


# ============================================================
# STEP 6 — EVALUATE THE MODEL
# ============================================================

def evaluate_model(model, X_test, y_test):
    print("\n📊 Evaluating model performance...")

    # -------------------------------------------------------
    # THRESHOLD SETTING
    # Default is 0.5 — lower it to catch more malaria cases
    # Example: 0.3 means "flag as Parasitized if 30% chance"
    # Lower value = catches more malaria but more false alarms
    # Higher value = fewer false alarms but misses more cases
    # -------------------------------------------------------
    THRESHOLD = 0.3   # ✏️ change this value (between 0.0 and 1.0)

    # Get probability of being Parasitized (index 1)
    probs       = model.predict_proba(X_test)[:, 1]

    # Apply threshold instead of default 0.5
    predictions = (probs >= THRESHOLD).astype(int)

    accuracy    = accuracy_score(y_test, predictions) * 100

    print(f"\n   ⚙️  Threshold used  : {THRESHOLD}")
    print(f"   🎯 Accuracy        : {accuracy:.2f}%")
    print("\n   Detailed Report:")
    print(classification_report(y_test, predictions,
                                target_names=["Uninfected", "Parasitized"]))

    # Count the 4 boxes
    cm             = confusion_matrix(y_test, predictions)
    true_negative  = cm[0][0]   # Correctly predicted Uninfected
    false_positive = cm[0][1]   # Uninfected but predicted Parasitized (false alarm)
    false_negative = cm[1][0]   # Parasitized but predicted Uninfected (dangerous!)
    true_positive  = cm[1][1]   # Correctly predicted Parasitized

    print("\n   Confusion Matrix Breakdown:")
    print(f"   ✅ Correctly found healthy   : {true_negative}")
    print(f"   ✅ Correctly found malaria   : {true_positive}")
    print(f"   ⚠️  Missed malaria cases     : {false_negative}  ← want this LOW")
    print(f"   ⚠️  False alarms             : {false_positive}")

    # --- Confusion Matrix Chart ---
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Uninfected", "Parasitized"],
                yticklabels=["Uninfected", "Parasitized"])
    plt.title(f"Confusion Matrix (Threshold = {THRESHOLD})")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.show()

    # --- Feature Importance ---
    features   = ["area_0", "area_1", "area_2", "area_3", "area_4"]
    importance = model.feature_importances_
    feat_df    = pd.DataFrame({"Feature": features, "Importance": importance})
    feat_df    = feat_df.sort_values("Importance", ascending=False)

    print("\n   Feature Importance Scores:")
    for _, row in feat_df.iterrows():
        print(f"   {row['Feature']}: {row['Importance']*100:.1f}%")

    plt.figure(figsize=(7, 4))
    plt.barh(feat_df["Feature"], feat_df["Importance"], color="#3498db")
    plt.xlabel("Importance Score")
    plt.title("Which Features Help Detect Malaria the Most?")
    plt.tight_layout()
    plt.show()


# ============================================================
# STEP 7 — PREDICT A NEW SAMPLE
# ============================================================

def predict_sample(model, scaler, le):
    """Ask one sample worth of values and predict."""
    features = ["area_0", "area_1", "area_2", "area_3", "area_4"]
    values   = []

    print("\nEnter the 5 area values from your blood sample.\n")
    for feature in features:
        while True:
            try:
                value = float(input(f"  Enter {feature}: "))
                values.append(value)
                break
            except ValueError:
                print("  ⚠️  Please enter a valid number.")

    sample     = np.array(values).reshape(1, -1)
    scaled     = scaler.transform(sample)
    result_num = model.predict(scaled)[0]
    prob       = model.predict_proba(scaled)[0]
    result     = le.inverse_transform([result_num])[0]   # back to text

    print("\n" + "-" * 40)
    if result == "Parasitized":
        print("  🔴 RESULT: PARASITIZED (Malaria Detected)")
        print(f"  Confidence: {prob[1] * 100:.1f}%")
        print("  ⚠️  Please consult a doctor immediately.")
    else:
        print("  🟢 RESULT: UNINFECTED (No Malaria)")
        print(f"  Confidence: {prob[0] * 100:.1f}%")
        print("  ✅ Blood sample appears healthy.")
    print("-" * 40)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 50)
    print("   MALARIA DETECTION WITH MACHINE LEARNING")
    print("=" * 50)

    df                                            = load_data()
    explore_data(df)
    visualize_data(df)
    X_train, X_test, y_train, y_test, scaler, le = prepare_data(df)
    model                                         = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

    # --- Ask user if they want to test their own sample ---
    while True:
        print("\n" + "=" * 50)
        print("  🔬 DO YOU WANT TO TEST YOUR OWN SAMPLE?")
        print("=" * 50)
        print("  [1] Yes — enter my blood sample values")
        print("  [2] No  — exit the program")
        print("-" * 50)

        answer = input("  Enter 1 or 2: ").strip()

        if answer == "1":
            predict_sample(model, scaler, le)

            # After prediction, ask if they want to test another
            again = input("\n  Test another sample? (yes / no): ").strip().lower()
            if again not in ["yes", "y"]:
                break

        elif answer == "2":
            break
        else:
            print("  ⚠️  Please enter 1 or 2.")

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
