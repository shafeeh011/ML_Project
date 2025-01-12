import os
import sys

from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_congig = ModelTrainerConfig()
        
        
        
    def initiate_model_trainer(self, train_array, test_array):
        try: 
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                #"Gradient Boosting": GradientBoostingRegressor(),
               # "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                #"AdaBoost Regressor": AdaBoostRegressor(),
                #"XGBRegressor": XGBRegressor()
            }
            
            model_report:dict = evaluate_model(X_train, y_train, X_test, y_test, models=models)
            
            best_model_score = max(sorted(model_report.values()))
            
            
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]
            
            if best_model_score < 0.6:
                raise CustomException("No best model found")
            
            logging.info("Best model found")
            
            save_object(
                file_path= self.model_trainer_congig.trained_model_path,
                obj=best_model
            )
            
            predicted = best_model.predict(X_test)
            
            r2_square = r2_score(y_test, predicted)
            
            return r2_square
        
        
        except Exception as e:
            raise CustomException(e, sys)