import os
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle

from src.exception import CustomException
from src.logger import logging


class DataTransformation:
    def __init__(self):
        self.preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Data Transformation started")

            # Read train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and Test data read successfully")

            # Split input and target
            input_feature_train = train_df.drop(columns=["salary"], axis=1)
            target_feature_train = train_df["salary"]

            input_feature_test = test_df.drop(columns=["salary"], axis=1)
            target_feature_test = test_df["salary"]

            logging.info("Input and target features separated")

            # Create preprocessing pipeline
            preprocessor = Pipeline(
                steps=[
                    ("scaler", StandardScaler())
                ]
            )

            # Fit on train and transform both
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train)
            input_feature_test_arr = preprocessor.transform(input_feature_test)

            logging.info("Data scaling completed")

            # Save preprocessor object
            os.makedirs(os.path.dirname(self.preprocessor_obj_file_path), exist_ok=True)

            with open(self.preprocessor_obj_file_path, "wb") as f:
                pickle.dump(preprocessor, f)

            logging.info("Preprocessor object saved successfully")

            return (
                input_feature_train_arr,
                input_feature_test_arr,
                target_feature_train,
                target_feature_test
            )

        except Exception as e:
            raise CustomException(e, sys)
