import os
import sys
from sklearn.ensemble import(AdaBoostRegressor,RandomForestRegressor,GradientBoostingRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj, evaluate_model




from dataclasses import dataclass

@dataclass

class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")


class ModelTrainer:
    def __init__(self):
        self.trained_model_config = ModelTrainerConfig()

    
    def initiate_model_trainer(self,train_arr,test_arr):
             try:

                    logging.info("Model training initiation started")
                    X_train,y_train,X_test,y_test = (
                        train_arr[:,:-1],train_arr[:,-1],test_arr[:,:-1],test_arr[:,-1]
                    )
                    
                    model_dict = {
                                            "Linear Regression": LinearRegression(),
                                            "Decision Tree": DecisionTreeRegressor(),
                                            "Random Forest": RandomForestRegressor(),
                                            "Gradient Boosting": GradientBoostingRegressor(),
                                            "AdaBoost": AdaBoostRegressor(),
                                            "K-Nearest Neighbors": KNeighborsRegressor(),
                                    }
                    model_report:dict = evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models = model_dict)
                
                    best_model_score = max(sorted(model_report.values()))
                    best_model_score_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
                    best_model = model_dict[best_model_score_name]
                    
                    if best_model_score < 0.6:
                            raise CustomException("No best model found")
                
                    save_obj(
                        file_path = self.trained_model_config.trained_model_file_path,
                        obj = best_model
                    )
                    logging.info(f"Found the best model :{best_model_score_name}")
                    prediction = best_model.predict(X_test)
                    model_r2_score = r2_score(y_test,prediction)

                    return model_r2_score
             except Exception as e:
                      raise CustomException(e,sys)