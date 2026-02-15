import os
from vehicle_insurance.constants import *
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

TIMESTAMP=datetime.now().strftime('%m_%d_%Y_%H_%M_%S')

@dataclass
class TrainingPipelineConfig:
    pipeline:str=PIPELINE
    artifact_dir:str=os.path.join(ARTIFACT_DIR,TIMESTAMP)
    timestamp:str=TIMESTAMP

training_pipeline:TrainingPipelineConfig=TrainingPipelineConfig()

@dataclass
class DataIngestionConfig:
    feature_store_dir:str=os.path.join(training_pipeline.artifact_dir,FEATURE_STORE_DIR)
    feature_store_file:str=os.path.join(feature_store_dir,raw_file)
    collection:str=COLLECTION
    test_split:float=TEST_SIZE
    ingested_dir:str=os.path.join(training_pipeline.artifact_dir,INGESTED_DIR)
    train_path:str=os.path.join(ingested_dir,train_file)
    test_path:str=os.path.join(ingested_dir,test_file)
    

@dataclass
class DataValidationConfig:
    data_validation_dir:str=os.path.join(training_pipeline.artifact_dir,DATA_VALIDATION_DIR)
    report_dir:str=os.path.join(data_validation_dir,DATA_VALIDATION_REPORT)

@dataclass
class DataTranformationConfig:
    model_pipeline_path:str=os.path.join(training_pipeline.artifact_dir,DATA_TRANSFORMED_DIR,preprocess_object)


@dataclass
class ModelTrainerConfig:
    model_save_path:str=os.path.join(training_pipeline.artifact_dir,MODEL_DIR,MODEL_SAVE_FILE)
    model_config_file:str=MODEL_CONFIG_FILE
    baseline_recall:float=MODEL_TRAINER_EXPECTED_RECALL

@dataclass
class ModelEvaluationConfig:

    changed_threshold_score:Optional[float]=MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE
    s3_bucket_name:str=S3_BUCKET_NAME
    s3_model_key_path: str = f"{BUCKET_S3_KEY}/{MODEL_SAVE_FILE}"

@dataclass
class ModelPusherConfig:
    
    s3_bucket_name:str=S3_BUCKET_NAME
    s3_model_key_path: str = f"{BUCKET_S3_KEY}/{MODEL_SAVE_FILE}"

@dataclass
class VehiclePredictorConfig:

    model_bucket_name:str=S3_BUCKET_NAME
    model_file_key_path:str=f"{BUCKET_S3_KEY}/{MODEL_SAVE_FILE}"
    



