
import pandas as pd
import numpy as np
import os

import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.components.model_evalution import ModelEvaluation
from src.components.data_ingestion import DataIngestion


from src.components.data_transformation import DataTransformation

from src.exception.exception import CustomException

from src.components.model_trainer import ModelTrainer



obj = DataIngestion()

train_data_path,test_data_path=obj.initiate_Data_Ingestion()


print(train_data_path)
print(test_data_path)
data_transformation=DataTransformation()
train_arr,test_arr=data_transformation.initiate_Data_Transformation(train_data_path,test_data_path)
print(train_arr)
print("hello g ")
model_trainer_obj=ModelTrainer()
model_trainer_obj.initiate_Model_trainer(train_arr,test_arr)

model_eval_obj = ModelEvaluation()
model_eval_obj.initiate_model_evaluation(train_arr,test_arr)



