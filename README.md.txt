# Loan Approval Prediction System

An end-to-end Machine Learning pipeline that predicts whether a loan application will be approved or rejected based on applicant financial and demographic data.

The trained model is deployed using a FastAPI REST API which allows users to send applicant information and receive real-time loan approval predictions.

---

## Project Overview

Financial institutions often need automated systems to evaluate loan applications quickly and efficiently.  
This project builds a machine learning model that predicts loan approval using historical applicant data.

The system includes:

- Data preprocessing
- Feature engineering
- Model training and evaluation
- Model serialization
- REST API deployment for real-time predictions

---

## Tech Stack

**Programming Language**

- Python

**Data Processing**

- Pandas  
- NumPy  

**Machine Learning**

- Scikit-learn

**Model Deployment**

- FastAPI  
- Uvicorn  
- Joblib  

**Development Tools**

- Jupyter Notebook  
- Swagger UI  

---

## Project Workflow

1. Data preprocessing and exploratory analysis performed in Jupyter Notebook.
2. Feature engineering applied to improve predictive performance.
3. Machine Learning pipeline built using Scikit-learn.
4. Multiple models were trained and evaluated.
5. Final model serialized using Joblib.
6. FastAPI REST API developed to serve predictions.
7. API tested using Swagger UI.

---

## Feature Engineering

To improve predictive performance, additional features were created:

**TotalIncome**

ApplicantIncome + CoapplicantIncome

**Income_to_Loan_Ratio**

TotalIncome / LoanAmount

**Is_High_Loan**

Binary indicator representing whether the loan amount is higher than the dataset median.

---

## Model Training

The following classification algorithms were evaluated:

- Logistic Regression
- Decision Tree
- Random Forest

Logistic Regression was selected for deployment because it provides stable performance, fast inference time, and easier interpretability.

**Model Accuracy:**  
Approximately **82% on the test dataset**

---

## Project Structure

loan-approval-prediction-ml

app.py                # FastAPI application  
train.py              # Model training pipeline  
requirements.txt      # Project dependencies  
README.md  

models/  
    loan_pipeline.pkl  

data/  
    loan_prediction.csv  

notebooks/  
    Loan_Approval_Prediction.ipynb  

assets/  
    api_documentation_swagger.png  
    predict_endpoint_overview.png  
    prediction_input_example.png  
    prediction_response_result.png  

---

## Installation

Clone the repository

git clone https://github.com/YOURUSERNAME/loan-approval-prediction-ml.git

Navigate to the project folder

cd loan-approval-prediction-ml

Install dependencies

pip install -r requirements.txt

---

## Train the Model

Run the training script

python train.py

This will train the machine learning pipeline and save the trained model inside the **models** folder.

---

## Run the API

Start the FastAPI server

uvicorn app:app --reload

Open the API documentation in your browser

http://127.0.0.1:8000/docs

---

## API Documentation

### Swagger API Interface

![Swagger Documentation](assets/api_documentation_swagger.png)

### Prediction Endpoint

![Predict Endpoint](assets/predict_endpoint_overview.png)

### Example Prediction Input

![Prediction Input](assets/prediction_input_example.png)

### Prediction Response

![Prediction Output](assets/prediction_response_result.png)

---

## Example API Request

Endpoint:

POST /predict

Example JSON input:

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

- Deploy the API on cloud platforms (AWS / Render / Railway)
- Build a frontend interface for loan prediction
- Implement advanced models such as XGBoost
- Add model monitoring and logging
- Containerize the application using Docker

---

## Author

Vinit Dave

AI & Data Science Graduate  
Software Developer | Aspiring Data Scientist / ML Engineer

---

If you found this project useful, consider giving it a star on GitHub.

