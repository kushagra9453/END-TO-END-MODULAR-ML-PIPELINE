import os
import sys
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging


class ModelTrainer:
    def __init__(self):
        self.model_file_path = os.path.join("artifacts", "model.pkl")

    def initiate_model_trainer(self, train_arr, test_arr, train_y, test_y):
        try:
            models = {
                "LinearRegression": LinearRegression(),
                "DecisionTree": DecisionTreeRegressor(),
                "RandomForest": RandomForestRegressor()
            }

            params = {
                "DecisionTree": {
                    "max_depth": [None, 5, 10]
                },
                "RandomForest": {
                    "n_estimators": [50, 100],
                    "max_depth": [None, 5]
                }
            }

            best_model = None
            best_score = -1

            for model_name, model in models.items():
                if model_name in params:
                    gs = GridSearchCV(model, params[model_name], cv=3)
                    gs.fit(train_arr, train_y)
                    model = gs.best_estimator_
                else:
                    model.fit(train_arr, train_y)

                preds = model.predict(test_arr)
                score = r2_score(test_y, preds)

                if score > best_score:
                    best_score = score
                    best_model = model

            with open(self.model_file_path, "wb") as f:
                pickle.dump(best_model, f)

            print("Best R2 Score:", best_score)
            return best_score

        except Exception as e:
            raise CustomException(e, sys)
