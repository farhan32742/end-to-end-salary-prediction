import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import mlflow
import mlflow.sklearn
import numpy as np
import pickle
from src.utils.utils import load_object
from urllib.parse import urlparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.logger.logging import logging
from src.exception.exception import CustomException

mlflow.set_experiment("mlops_with_sunny")
#below line is used to set the tracking uri for local mlflow server
#mlflow.set_tracking_uri("https://127.0.0.1:5000")
class ModelEvaluation:
    def __init__(self):
        logging.info("Evaluation started")
    
    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))  # RMSE
        mae = mean_absolute_error(actual, pred)  # MAE
        r2 = r2_score(actual, pred)  # R² Score
        logging.info("Evaluation metrics captured")
        return rmse, mae, r2

    def initiate_model_evaluation(self, train_array, test_array):
        try:
            X_test, y_test = test_array[:, :-1], test_array[:, -1]
            
            model_path = os.path.join("artifacts", "model.pkl")
            model = load_object(model_path)
            
            logging.info("Model has been loaded successfully")
            
            #tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            #logging.info(f"MLflow Tracking URI Type: {tracking_url_type_store}")
            


            with mlflow.start_run():
                predictions = model.predict(X_test)
                rmse, mae, r2 = self.eval_metrics(y_test, predictions)
                
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("r2", r2)
                mlflow.log_metric("mae", mae)
                
                example_input = X_test[:1]  # Providing an example input for MLflow
                mlflow.sklearn.log_model(model, "model",input_example=example_input)
        
                # Get the run_id
                #run_id = run.info.run_id
        
                # Generate the model_uri
               # model_uri = f"runs:/{run_id}/model"
        
                 # Register the model
                #mlflow.register_model(model_uri, "ml_model")


                
                logging.info("Model evaluation completed and logged in MLflow")
        
        except Exception as e:
            logging.error(f"Error during model evaluation: {str(e)}")
            raise CustomException(e, sys)
