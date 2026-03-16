# ==========================================
#   SIMPLE MALARIA DETECTION WITH ML
# ==========================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


# ------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------

df = pd.read_csv("dataset.csv")


# ------------------------------------------
# 2. DATA PREPROCESSING
# ------------------------------------------

# Convert labels to numbers
le = LabelEncoder()
df["Label"] = le.fit_transform(df["Label"])

# Features and target
X = df[["area_0", "area_1", "area_2", "area_3", "area_4"]]
y = df["Label"]


# ------------------------------------------
# 3. TRAIN MODEL
# ------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier()

model.fit(X_train, y_train)


# ------------------------------------------
# 4. EVALUATE MODEL
# ------------------------------------------

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", round(accuracy * 100, 2), "%")


# ------------------------------------------
# 5. PREDICT MALARIA FROM NEW SAMPLE
# ------------------------------------------

print("\nEnter blood cell feature values:")

area_0 = float(input("area_0: "))
area_1 = float(input("area_1: "))
area_2 = float(input("area_2: "))
area_3 = float(input("area_3: "))
area_4 = float(input("area_4: "))

sample = pd.DataFrame(
    [[area_0, area_1, area_2, area_3, area_4]],columns=["area_0", "area_1", "area_2", "area_3", "area_4"]
)

prediction = model.predict(sample)

result = le.inverse_transform(prediction)

print("\nPrediction:", result[0])
