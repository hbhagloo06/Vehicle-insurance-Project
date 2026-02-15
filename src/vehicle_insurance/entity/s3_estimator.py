from vehicle_insurance.cloud_storage.aws_storage import SimpleStorageService
from vehicle_insurance.exception import MyException
import sys
from pandas import DataFrame
from vehicle_insurance.entity.estimator import MyModel


class Proj1Estimator:
    def __init__(self,model_path:str,bucket_name:str):
        self.bucket_name=bucket_name
        self.model_path=model_path
        self.s3=SimpleStorageService()
        self.model:MyModel=None


    def is_model_present(self)->bool:
        try:
            return self.s3.s3_key_path_available(self.bucket_name,self.model_path)
        except Exception as e:
            raise MyException(e,sys) from e
    
    def load_model(self)->MyModel:
        return self.s3.load_model(self.model_path,self.bucket_name)
    
    def save_model(self,from_filename:str,remove:bool=False):
        try:
            self.s3.upload_file(from_filename,self.model_path,self.bucket_name,remove=remove)
        except Exception as e:
            raise MyException(e,sys) from e
        
    def predict(self, data_frame: DataFrame):
        try:
            if self.model is None:
                if not self.is_model_present:
                    raise FileNotFoundError("Model not present")
                self.model = self.load_model()

            return self.model.predict(data_frame)

        except Exception as e:
            raise MyException(e, sys) from e
    
    def predict_with_confidence_score(self, data_frame: DataFrame):
        try:
            if self.model is None:
                if not self.is_model_present:
                    raise FileNotFoundError("Model not present")
                self.model = self.load_model()

            return self.model.predict_with_confidence_score(data_frame)

        except Exception as e:
            raise MyException(e, sys) from e
    
