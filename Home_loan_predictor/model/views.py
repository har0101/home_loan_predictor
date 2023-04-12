from django.shortcuts import render
from django.urls import path

# Create your views here.
from joblib import load
from sklearn.preprocessing import StandardScaler
import pandas as pd


model = load('./savedmodel/loan_predictor.joblib')
scaler = load('./savedmodel/scaled.joblib')


def predictor(request):
    return render(request,'home_loan_predictor.html')

def take_info(request):
    Gender = request.POST['Gender']
    Married = request.POST['Married']
    Dependents = request.POST['Dependents']
    Education = request.POST['Education']
    Self_Employed = request.POST['Self_Employed']
    ApplicantIncome = request.POST['ApplicantIncome']
    CoapplicantIncome = request.POST['CoapplicantIncome']
    LoanAmount = request.POST['LoanAmount']
    Loan_Amount_Term = request.POST['Loan_Amount_Term']
    Credit_History = request.POST['Credit_History']
    Property_Area = request.POST['Property_Area']


    # scaler = StandardScaler()
    y_pred =model.predict(scaler.transform([[Gender, Married, Dependents, Education,
       Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount,
       Loan_Amount_Term, Credit_History, Property_Area]])) 
    # y_df = pd.DataFrame({'Gender':Gender,'Married':Married,'Dependents':Dependents,'Education':Education,'Self_Employed':Self_Employed,'ApplicantIncome':ApplicantIncome,'CoapplicantIncome':CoapplicantIncome,'LoanAmount':LoanAmount,'Loan_Amount_Term':Loan_Amount_Term,'Credit_History':Credit_History,'Property_Area':Property_Area},index=[0])
    # print(y_df)
    # # y_final_df = scaler.fit_transform(y_df)
    # print(y_final_df)
    # y_pred = model.predict((y_final_df))
    print(y_pred)
    # print(LoanAmount)

    if y_pred[0]==0:
        # k = '"NO" please consult to manager'
        k = '"NO" please consult to manager'
    else:
        k = '"yes" you can take home loan please process further'


    return render(request,'result.html',{'result':k})
