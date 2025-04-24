import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import pandas as pd
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from src.utils.utils import save_object
from src.logger.logging import logging
from src.exception.exception import CustomException


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_transformation_data(self):
        try:
            categorical_cols = ['SEX', 'DESIGNATION', 'UNIT']
            numerical_cols = ['AGE', 'PAST EXP', 'Total_time_in_company']

            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OrdinalEncoder()),
                ("scaler", StandardScaler())
            ])

            preprocessor = ColumnTransformer(transformers=[
                ("num_pipeline", num_pipeline, numerical_cols),
                ("cat_pipeline", cat_pipeline, categorical_cols)
            ])

            return preprocessor

        except Exception as e:
            logging.info("Error occurred while building preprocessing object")
            raise CustomException(e, sys)

    def initiate_Data_Transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Train and test data loaded successfully")

            # Convert dates and calculate total time in company
            for df in [train_df, test_df]:
                df.drop(columns=['FIRST NAME', 'LAST NAME', 'LEAVES USED'], inplace=True)
                df['DOJ'] = pd.to_datetime(df['DOJ'], errors='coerce')
                df['CURRENT DATE'] = pd.to_datetime(df['CURRENT DATE'], errors='coerce')

                 # Remove rows where either date is missing
                df.dropna(subset=['DOJ', 'CURRENT DATE'], inplace=True)

                # Ensure 'Total_time_in_company' is created in place for both train and test DataFrames
                df['Total_time_in_company'] = ((df['CURRENT DATE'] - df['DOJ']).dt.days / 30).astype(int)
    
            preprocessing_obj = self.get_transformation_data()
            target_column_name = 'SALARY'

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            print("Train columns:", input_feature_train_df.columns.tolist())

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applied preprocessing on training and testing data")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("Preprocessing object saved successfully")

            return (
                train_arr,
                test_arr
            )

        except Exception as e:
            logging.info("Error during data transformation")
            raise CustomException(e, sys)
