import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation


class DataIngestion:
    def __init__(self):
        self.raw_data_path = os.path.join("artifacts", "data.csv")
        self.train_data_path = os.path.join("artifacts", "train.csv")
        self.test_data_path = os.path.join("artifacts", "test.csv")

    def initiate_data_ingestion(self):
        logging.info("Entered data ingestion method")

        try:
            df = pd.read_csv("notebook/data.csv")
            logging.info("Read dataset as dataframe")

            os.makedirs(os.path.dirname(self.raw_data_path), exist_ok=True)

            df.to_csv(self.raw_data_path, index=False)

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.train_data_path, index=False)
            test_set.to_csv(self.test_data_path, index=False)

            logging.info("Train and Test data saved")

            return self.train_data_path, self.test_data_path

        except Exception as e:
            raise CustomException(e, sys)


from src.components.model_trainer import ModelTrainer

if __name__ == "__main__":
    ingestion_obj = DataIngestion()
    train_path, test_path = ingestion_obj.initiate_data_ingestion()

    transformation_obj = DataTransformation()
    train_arr, test_arr, train_y, test_y = transformation_obj.initiate_data_transformation(train_path, test_path)

    model_trainer_obj = ModelTrainer()
    print(model_trainer_obj.initiate_model_trainer(train_arr, test_arr, train_y, test_y))
