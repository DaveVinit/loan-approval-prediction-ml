import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# =========================
# 1️⃣ Load Dataset
# =========================
df = pd.read_csv("loan_prediction.csv")

# Drop missing values
df.dropna(inplace=True)


# =========================
# 2️⃣ Feature Engineering
# =========================
df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
df["Income_to_Loan_Ratio"] = df["TotalIncome"] / df["LoanAmount"]
df["Is_High_Loan"] = (df["LoanAmount"] > df["LoanAmount"].median()).astype(int)

# Encode target
df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})


# =========================
# 3️⃣ Define Features & Target
# =========================
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]


# =========================
# 4️⃣ Column Definitions
# =========================
numerical_cols = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "TotalIncome",
    "Income_to_Loan_Ratio",
    "Is_High_Loan"
]

categorical_cols = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area"
]


# =========================
# 5️⃣ Preprocessing
# =========================
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(drop="first"), categorical_cols)
    ]
)


# =========================
# 6️⃣ Full Pipeline
# =========================
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LogisticRegression())
])


# =========================
# 7️⃣ Train Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =========================
# 8️⃣ Train Model
# =========================
pipeline.fit(X_train, y_train)


# =========================
# 9️⃣ Evaluate
# =========================
y_pred = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))


# =========================
# 🔟 Save Model + Median
# =========================
loan_median = df["LoanAmount"].median()

joblib.dump(
    {
        "pipeline": pipeline,
        "loan_median": loan_median
    },
    "models/loan_pipeline.pkl"
)

print("Model and median saved successfully!")
