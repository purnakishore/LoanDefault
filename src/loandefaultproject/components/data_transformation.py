import os
import sys
from dataclasses import dataclass

from src.loandefaultproject.logger import logging
from src.loandefaultproject.exception import CustomException

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings(action='ignore')

from imblearn.over_sampling import SMOTE

@dataclass
class DataTransformationConfig:
    ml_data_path:str = os.path.join('artifacts','ml_data.csv')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def initiate_data_transformations(self,cleaned_data_path):
        df = pd.read_csv(cleaned_data_path)

        df_nums = df.select_dtypes(exclude='object')
        df_objs = df.select_dtypes(include='object')

        df_objs = pd.get_dummies(df_objs,drop_first=True)

        final_df = pd.concat([df_nums,df_objs],axis=1)

        X = final_df.drop('Default',axis=1)
        y = final_df['Default']

        smote = SMOTE(sampling_strategy='minority')
        X_sm, y_sm = smote.fit_resample(X, y)

        ml_data = pd.concat([X_sm,y_sm],axis=1)

        ml_data.to_csv('artifacts\ml_df.csv')

        return self.data_transformation_config.ml_data_path



