import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from src.logger import logging
from src.exception import CustomException
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import os
from src.utils import save_obj
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
     preprocessor_obj_file_path = os.path.join("artifacts","preprocessor.pkl")


class DataTransformation:
     def __init__(self):
            self.data_transformation_config = DataTransformationConfig()
     def get_data_transformation(self):
            try:
               numerical_features = ['reading_score', 'writing_score']
               categorical_features = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

               num_pipeline = Pipeline(
                    steps = [('imputer',SimpleImputer(strategy="median")),('scaler',StandardScaler())]
               )

               cat_pipeline = Pipeline(
                      steps = [('imputer',SimpleImputer(strategy="most_frequent")),
                               ('Onehot encoder',OneHotEncoder()),
                               ('scaler',StandardScaler(with_mean=False))]
               )

               logging.info("numerical and column data transformation completed")

               preprocessor = ColumnTransformer([
                    ("num_pipelines",num_pipeline,numerical_features),
                    ("cat_pipelines",cat_pipeline,categorical_features)
                    ]
               )

               return preprocessor
   
            except Exception as e:
                raise CustomException(e,sys)
            

     def initiate_data_transformation(self,train_path,test_path):
        try:
                train_df = pd.read_csv(train_path)
                test_df = pd.read_csv(test_path)

                logging.info("Train and test data reading completed")
                logging.info("Obtaining preprocessing object")
                preprocessing_obj = self.get_data_transformation()
                target_column_name = "math_score"

                input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
                target_feature_train_df = train_df[target_column_name]

                input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
                target_feature_test_df = test_df[target_column_name]

                logging.info(f"Applying preprocessing obj on training and testing dataframe")

                input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
                input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

                train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
                test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]


                save_obj(
                      file_path = self.data_transformation_config.preprocessor_obj_file_path,
                      obj = preprocessing_obj
                )

                return(
                      train_arr,
                      test_arr,
                      self.data_transformation_config.preprocessor_obj_file_path
                )
        except Exception as e:
            raise CustomException(e,sys)


          
