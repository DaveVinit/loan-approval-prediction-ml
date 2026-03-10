# Loan Approval Prediction System

A Machine Learning project that predicts whether a loan application will be approved or rejected based on applicant financial and demographic information.
The project includes model training, feature engineering, and deployment through a FastAPI REST API for real-time predictions.

---

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, FastAPI, Joblib

---

## Project Workflow

1. Data preprocessing and exploratory analysis performed in Jupyter Notebook.
2. Feature engineering applied to improve predictive performance.
3. Machine Learning pipeline built using Scikit-learn with preprocessing and model training steps.
4. Model serialized using Joblib.
5. FastAPI REST API created to serve predictions.
6. API tested using Swagger UI.

---

## Feature Engineering

The following features were engineered to improve model performance:

* **TotalIncome** = ApplicantIncome + CoapplicantIncome
* **Income_to_Loan_Ratio** = TotalIncome / LoanAmount
* **Is_High_Loan** = Indicator if loan amount is above dataset median

---

## Model Training

Multiple models were evaluated:

* Logistic Regression
* Decision Tree
* Random Forest

Logistic Regression was selected for deployment due to its simplicity, efficiency, and reliable performance.

Model accuracy on the test dataset: **~82%**

---

## Project Structure

loan-approval-prediction-ml
│
├── app.py              # FastAPI deployment
├── train.py            # Model training pipeline
├── requirements.txt    # Project dependencies
├── README.md
│
├── models
│   └── loan_pipeline.pkl
│
├── data
│   └── loan_prediction.csv
│
└── notebooks
└── Loan_Approval_Prediction.ipynb

---

## Installation

Clone the repository:

git clone [https://github.com/YOURUSERNAME/loan-approval-prediction-ml.git](https://github.com/YOURUSERNAME/loan-approval-prediction-ml.git)

Navigate to the project folder:

cd loan-approval-prediction-ml

Install dependencies:

pip install -r requirements.txt

---

## Train the Model

python train.py

---

## Run the API

uvicorn app:app --reload

Open API documentation:

[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## Example API Request

POST /predict

{
"ApplicantIncome": 5000,
"CoapplicantIncome": 2000,
"LoanAmount": 150,
"Loan_Amount_Term": 360,
"Credit_History": 1,
"Gender": "Male",
"Married": "Yes",
"Dependents": "0",
"Education": "Graduate",
"Self_Employed": "No",
"Property_Area": "Urban"
}

---

## Future Improvements

* Add model performance comparison dashboard
* Deploy API to cloud platform (Render / AWS / Railway)
* Build a simple frontend interface for predictions
