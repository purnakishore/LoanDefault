import os
import sys
from src.loandefaultproject.logger import logging
from src.loandefaultproject.exception import CustomException
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings(action='ignore')

from src.loandefaultproject.utils import year_of,amount_object,undefine_demogr

from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    cleaned_data_path:str = os.path.join('artifacts','cleaned_data.csv')
    raw_data_path:str = os.path.join('artifacts','DS_Interview_Virtual_Task_training_data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        df = pd.read_csv(self.ingestion_config.raw_data_path,low_memory=False,parse_dates=['Date_Of_Disbursement','Commitment_Date'])

        revolve_map = {'No':'No','0':'No','Yes':'Yes','T':'T','R':'R','`':np.nan,'1':'1','2':'2','4':'4','.':np.nan,'C':'C'}
        df['Revolving_Credit_Line'] =df['Revolving_Credit_Line'].map(revolve_map)
        df['Revolving_Credit_Line'].value_counts()

        low_map = {'No':'No','Yes':'Yes','0':'No','S':'S','C':'C','R':'R','A':'A'}
        df['Low_Documentation_Loan'] = df['Low_Documentation_Loan'].map(low_map)
        df['Low_Documentation_Loan'].map(low_map).value_counts()

        Not_usefull_columns = ['ID','Classification_Code ','Primary_Loan_Digit']
        df.drop(Not_usefull_columns,axis=1,inplace=True)

        df.dropna(axis=0,inplace=True)

        
            
        df['Year_Of_Commitment ']=df['Year_Of_Commitment '].apply(year_of)

        
        
        df['Guaranteed_Approved _Loan'] = df['Guaranteed_Approved _Loan'].apply(amount_object)
        df['ChargedOff_Amount '] = df['ChargedOff_Amount '].apply(amount_object)
        df['Gross_Amount_Balance'] = df['Gross_Amount_Balance'].apply(amount_object)
        df['Loan_Approved_Gross'] = df['Loan_Approved_Gross'].apply(amount_object)
        df['Gross_Amount_Disbursed  '] = df['Gross_Amount_Disbursed  '].apply(amount_object)

        df.drop('Code_Franchise',axis=1,inplace=True)

        df['Disbursement_day'] = df['Date_Of_Disbursement'].dt.day
        df['Disbursement_month'] = df['Date_Of_Disbursement'].dt.month
        df['Disbursement_year'] = df['Date_Of_Disbursement'].dt.year
        df.drop('Date_Of_Disbursement',axis=1,inplace=True)

        df['Commitment_day'] = df['Commitment_Date'].dt.day
        df['Commitment_month'] = df['Commitment_Date'].dt.month
        df['Commitment_year'] = df['Commitment_Date'].dt.year
        df.drop('Commitment_Date',axis=1,inplace=True)

        df.to_csv('artifacts/clean_data.csv')
        
        df['demograp'] = np.vectorize(undefine_demogr)(df['Borrower_City'],df['Demography'])

        df.drop(['Borrower_Name ','Borrower_City','Demography'],axis=1,inplace=True)

        df.rename(columns={'demograp':'Demography'},inplace=True)

        df.to_csv(self.ingestion_config.cleaned_data_path,index=False)

        return self.ingestion_config.cleaned_data_path
