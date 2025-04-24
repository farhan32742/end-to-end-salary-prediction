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
            numerical_cols = ['AGE', 'EXPERIENCE','Total_time_in_company']

            # No custom order provided, OrdinalEncoder will infer from data
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
            logging.error("Error in get_transformation_data")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, raw_data_path: str):
        try:
            df = pd.read_csv(raw_data_path)
            logging.info("Data loaded successfully")

            df.drop(columns=['FIRST NAME', 'LAST NAME','LEAVES USED'], inplace=True)
            df['DOJ'] = pd.to_datetime(df['DOJ'])
            df['CURRENT DATE'] = pd.to_datetime(df['CURRENT DATE'])
            df['Total_time_in_company'] = ((df['CURRENT DATE'] - df['DOJ']).dt.days / 30).astype(int)
            df.drop(columns=['DOJ', 'CURRENT DATE'], inplace=True)

            preprocessing_obj = self.get_transformation_data()

            target_column_name = 'SALARY'
            input_df = df.drop(columns=[target_column_name])
            target_df = df[target_column_name]

            input_arr = preprocessing_obj.fit_transform(input_df)

            full_arr = np.c_[input_arr, np.array(target_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("Preprocessor object saved successfully")

            return full_arr

        except Exception as e:
            logging.error("Error in initiate_data_transformation")
            raise CustomException(e, sys)
