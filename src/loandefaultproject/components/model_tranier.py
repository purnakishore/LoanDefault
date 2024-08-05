import os
import sys
from dataclasses import dataclass
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,f1_score

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings(action='ignore')

from src.loandefaultproject.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,ml_data_path):

        df = pd.read_csv(ml_data_path)

        X = df.drop('Default',axis=1)
        y = df['Default']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101,stratify=y)
        
        scaler = StandardScaler()

        scaled_X_train = scaler.fit_transform(X_train)
        scaled_X_test = scaler.transform(X_test)

        model = LogisticRegression()

        model.fit(scaled_X_train,y_train)

        y_pred = model.predict(scaled_X_test)

        accuracy = accuracy_score(y_test,y_pred)
        f1_score = f1_score(y_test,y_pred)

        # confusion_matrix(y_test,y_pred)

        # print(classification_report(y_test,y_pred))
        
        mlflow.set_registry_uri("https://dagshub.com/krishnaik06/mlprojecthindi.mlflow")
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():

                predicted_qualities = model.predict(X_test)

                mlflow.log_metric("accuracy",accuracy)
                mlflow.log_metric("f1_score", f1_score)


                # Model registry does not work with file store
                if tracking_url_type_store != "file":

                    # Register the model
                    # There are other ways to use the Model Registry, which depends on the use case,
                    # please refer to the doc for more information:
                    # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                    mlflow.sklearn.log_model(model, "model", registered_model_name=model)
                else:
                    mlflow.sklearn.log_model(model, "model")

        save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )