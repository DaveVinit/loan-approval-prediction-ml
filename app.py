from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Loan Approval Prediction API")

# Load pipeline
#model = joblib.load("models/loan_pipeline.pkl")

saved_obj = joblib.load("models/loan_pipeline.pkl")
model = saved_obj["pipeline"]
loan_median = saved_obj["loan_median"]


# Request schema
class LoanData(BaseModel):
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: float
    Gender: str
    Married: str
    Dependents: str
    Education: str
    Self_Employed: str
    Property_Area: str

@app.get("/")
def home():
    return {"message": "Loan Prediction API Running Successfully"}

@app.post("/predict")
def predict(data: LoanData):

    input_df = pd.DataFrame([data.dict()])

    input_df["TotalIncome"] = input_df["ApplicantIncome"] + input_df["CoapplicantIncome"]
    input_df["Income_to_Loan_Ratio"] = input_df["TotalIncome"] / input_df["LoanAmount"]
    input_df["Is_High_Loan"] = (
        input_df["LoanAmount"] > loan_median
    ).astype(int)

    prediction = model.predict(input_df)[0]

    result = "Loan Approved" if prediction == 1 else "Loan Rejected"

    return {"prediction": result}
