import pickle
import os
import pandas as pd


class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

    def predict(self, features):
        with open(self.model_path, "rb") as f:
            model = pickle.load(f)

        with open(self.preprocessor_path, "rb") as f:
            preprocessor = pickle.load(f)

        data_scaled = preprocessor.transform(features)
        prediction = model.predict(data_scaled)

        return prediction
