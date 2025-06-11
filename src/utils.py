import numpy as np 
import sys
import pandas as pd
import os
from src.exception import CustomException
import dill
# from catboost import CatBoostRegressor
from sklearn.ensemble import(AdaBoostRegressor,RandomForestRegressor,GradientBoostingRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
# from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import GridSearchCV


from dataclasses import dataclass

def save_obj(file_path,obj):
    try:
        dir_name = os.path.dirname(file_path)
        os.makedirs(dir_name,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(sys,e)
    
def evaluate_model(X_train,y_train,X_test,y_test,models,params):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]
            gs = GridSearchCV(model,param,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            train_model_score = r2_score(y_train,y_train_pred)
            test_model_score  = r2_score(y_test,y_test_pred)

            model_names = list(models.keys())
            report[model_names[i]] = test_model_score
        return report
    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
        try:

            with open(file_path,'rb') as file_obj:
                logging.info('Loading of pickle file happened succesfully')
                return dill.load(file_obj)
        except Exception as e:
            raise CustomException(e,sys)