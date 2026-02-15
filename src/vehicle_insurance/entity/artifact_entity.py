from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    train_path:str
    test_path:str

@dataclass
class DataValidationArtifact:
    validation_status:bool
    message:str
    report_path:str

@dataclass
class DataTransformationArtifact:
    model_pipeline_path:str

@dataclass
class ClassificationMetricsArtifact:
    f1:float
    roc_auc:float
    recall:float
    average_precision:float

@dataclass
class ModelTrainerArtifact:
    trained_model_path:str
    metrics:ClassificationMetricsArtifact

@dataclass
class ModelEvaluationArtifact:
    is_model_accepted:bool
    diff_f1:float   #How much improvement (or degradation) happened.
    s3_model_path:str
    trained_model_path:str

@dataclass
class ModelPusherArtifact:
    s3_model_path:str
    bucket_name:str