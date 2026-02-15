from vehicle_insurance.entity.config_entity import ModelPusherConfig
from vehicle_insurance.entity.s3_estimator import SimpleStorageService
from vehicle_insurance.entity.s3_estimator import Proj1Estimator
from vehicle_insurance.entity.artifact_entity import ModelEvaluationArtifact,ModelPusherArtifact
from vehicle_insurance.exception import MyException
import sys
import logging

logger=logging.getLogger('ModelPusher')

class ModelPusher:
    def __init__(self,model_pusher_config:ModelPusherConfig,s3:SimpleStorageService,model_eval_artifact:ModelEvaluationArtifact):
        self.model_pusher_config=model_pusher_config
        self.bucket_name=self.model_pusher_config.s3_bucket_name
        self.model_key_path=self.model_pusher_config.s3_model_key_path
        self.s3=s3
        self.proj1_estimator=Proj1Estimator(model_path=self.model_key_path,bucket_name=self.bucket_name)
        self.model_eval_artifact=model_eval_artifact


    def initiate_model_pusher(self)->ModelPusherArtifact:
        logger.info("Entered initiate_model_pusher method of ModelTrainer class")
        
        try:
            print("------------------------------------------------------------------------------------------------")
            logger.info("Uploading artifacts folder to s3 bucket")
            
            logger.info("Uploading new model to S3 bucket....")
            self.proj1_estimator.save_model(from_filename=self.model_eval_artifact.trained_model_path,remove=False)
            model_pusher_artifact=ModelPusherArtifact(s3_model_path=self.model_key_path,bucket_name=self.bucket_name)

            logger.info("Uploaded artifacts folder to s3 bucket")
            logger.info(f"Model pusher artifact: [{model_pusher_artifact}]")
            logger.info("Exited initiate_model_pusher method of ModelTrainer class")

            return model_pusher_artifact
        
        except Exception as e:
            raise MyException(e,sys) from e


