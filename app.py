from src.loandefaultproject.logger import logging
from src.loandefaultproject.exception import CustomException
from src.loandefaultproject.components.data_ingestion import DataIngestion
import sys


if __name__ == "__main__":
    logging.info("The Execution has started")

    try:
        data_ingestion=DataIngestion()
        cleaned_data_path=data_ingestion.initiate_data_ingestion()
        
    except Exception as e:
        logging.info("custom exception")
        raise CustomException(e,sys)