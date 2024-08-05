import os
import sys
from src.loandefaultproject.logger import logging
from src.loandefaultproject.exception import CustomException
import pandas as pd
import numpy as np
import warnings


def year_of(year):
    try:
        if np.int64(year):
            return np.int64(year)
    except:
        return np.int64(year[:-1])
    

def amount_object(str):
    return np.float64(str.replace('Rs.',''))

df = pd.read_csv('artifacts/clean_data.csv')

def undefine_demogr(borower_city,demography):
    if demography == 'Undefined':
        return df[df['Borrower_City'] == borower_city]['Demography'].value_counts().reset_index().query('Demography != "Undefined"').sort_values('count',ascending=False)['Demography'].iloc[0]
    else:
        return demography