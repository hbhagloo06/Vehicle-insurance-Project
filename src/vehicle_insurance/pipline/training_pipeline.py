from vehicle_insurance.components.data_ingestion import DataIngestion
from vehicle_insurance.components.data_validation import DataValidation
from vehicle_insurance.components.model_evaluation import ModelEvaluation
from vehicle_insurance.components.data_transformation import DataTransformation
from vehicle_insurance.entity.config_entity import DataIngestionConfig,DataValidationConfig,DataTranformationConfig,ModelTrainerConfig,ModelEvaluationConfig,ModelPusherConfig
from vehicle_insurance.exception import MyException
from vehicle_insurance.components.model_trainer import ModelTrainer
from vehicle_insurance.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact,DataTransformationArtifact,ModelTrainerArtifact,ModelEvaluationArtifact
import sys
from vehicle_insurance.entity.s3_estimator import SimpleStorageService
from vehicle_insurance.components.model_pusher import ModelPusher
import logging

logger=logging.getLogger(__name__)
class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config=DataIngestionConfig()
        self.data_validation_config=DataValidationConfig()
        self.data_transformation_config=DataTranformationConfig()
        self.model_trainer_config=ModelTrainerConfig()
        self.model_eval_config=ModelEvaluationConfig()
        self.model_pusher_config=ModelPusherConfig()
        self.s3=SimpleStorageService()

    def data_ingestion(self)->DataIngestionArtifact:
        try:
            data_ingestion=DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact=data_ingestion.initiate_data_ingestion()
            logger.info('sucessfully got the train and test set')
            logger.info('exiting data_ingestion')
            return data_ingestion_artifact
        
        except Exception as e:
            raise MyException(e,sys) from e
        
    def data_validation(self,data_ingestion_artifact:DataIngestionArtifact)->DataValidationArtifact:
        try:
            data_validation=DataValidation(data_ingestion_artifact=data_ingestion_artifact,
                                           data_validation_config=self.data_validation_config)
            
            data_validation_artefacts=data_validation.initiate_validation()
            logger.info('exiting data validation')
            return data_validation_artefacts
        except Exception as e:
            raise MyException(e,sys) from e
        
    def data_transformation(self,data_validation_artifact:DataValidationArtifact,data_ingestion_artifact:DataIngestionArtifact):
        try:
            data_transformed=DataTransformation(data_validation_artifact=data_validation_artifact,
                               data_ingestion_artifact=data_ingestion_artifact,
                               data_transformation_config=self.data_transformation_config)
            data_transformation_artifact=data_transformed.initiate_data_transformation()
            logger.info('exiting data transformation')
            return data_transformation_artifact

        except Exception as e:
            raise MyException(e,sys) from e
        
    def model_trainer(self,data_transformation_artifact:DataTransformationArtifact,model_trainer_config:ModelTrainerConfig,data_ingestion_artifact:DataIngestionArtifact):
        try:
            model=ModelTrainer(data_transformation_artifact=data_transformation_artifact,
                         model_trainer_config=model_trainer_config,data_ingestion_artifact=data_ingestion_artifact)
            model_trainer_artifact=model.initiate_model_trainer()
            logger.info('exiting model Training')
            return model_trainer_artifact

        except Exception as e:
            raise MyException(e,sys) from e
        
    def model_evaluation(self,model_trainer_artifact: ModelTrainerArtifact, data_ingestion_artifact: DataIngestionArtifact, model_eval_config: ModelEvaluationConfig):
        try:
            model_eval=ModelEvaluation(model_trainer_artifact=model_trainer_artifact,
                            data_ingestion_artifact=data_ingestion_artifact,
                            model_eval_config=model_eval_config)
            model_eval_artifact=model_eval.evaluate_model()
            logger.info('exiting model evaluation')
            return model_eval_artifact
        except Exception as e:
            raise MyException(e,sys) from e
        
    def model_pusher(self,model_pusher_config: ModelPusherConfig, s3: SimpleStorageService, model_eval_artifact: ModelEvaluationArtifact):
        try:
            model_pusher=ModelPusher(model_pusher_config=model_pusher_config,
                        s3=s3,
                        model_eval_artifact=model_eval_artifact)
            model_pusher_artifact=model_pusher.initiate_model_pusher()
            return model_pusher_artifact
        
        except Exception as e:
            raise MyException(e,sys) from e

        
    def run_pipeline(self):
        try:
            data_ingestion_artifact=self.data_ingestion()
            data_validation_artifact=self.data_validation(data_ingestion_artifact)
            data_transformation_artifact=self.data_transformation(data_validation_artifact,data_ingestion_artifact)
            model_trainer_artifact=self.model_trainer(data_transformation_artifact,self.model_trainer_config,data_ingestion_artifact)
            model_eval_artifact=self.model_evaluation(model_trainer_artifact,data_ingestion_artifact,self.model_eval_config)
            if not model_eval_artifact.is_model_accepted:
                logger.info('The model is not suitable to be pushed to production.It has a lower F1 Score than the existing model')
                return
            model_pusher_artifact=self.model_pusher(self.model_pusher_config,self.s3,model_eval_artifact)

        except Exception as e:
            raise MyException(e,sys) from e
