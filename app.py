from src.loandefaultproject.logger import logging
from src.loandefaultproject.exception import CustomException
from src.loandefaultproject.components.data_ingestion import DataIngestion
from src.loandefaultproject.components.data_transformation import DataTransformation
from src.loandefaultproject.components.model_tranier import ModelTrainer
import sys


if __name__ == "__main__":
    logging.info("The Execution has started")

    try:
        data_ingestion=DataIngestion()
        cleaned_data_path=data_ingestion.initiate_data_ingestion()

        data_transformation = DataTransformation()
        ml_data = data_transformation.initiate_data_transformations(cleaned_data_path)

        model_trainer = ModelTrainer()
        print(model_trainer.initiate_model_trainer(ml_data))


    except Exception as e:
        logging.info("custom exception")
        raise CustomException(e,sys)