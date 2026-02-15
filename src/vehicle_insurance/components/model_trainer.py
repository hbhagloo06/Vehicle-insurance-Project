import logging
from vehicle_insurance.exception import MyException
from vehicle_insurance.entity.artifact_entity import DataIngestionArtifact,DataTransformationArtifact,ClassificationMetricsArtifact,ModelTrainerArtifact
from vehicle_insurance.utils.main_utils import read_yaml_file,load_object,save_object
from vehicle_insurance.constants import *
from sklearn.metrics import average_precision_score,recall_score,f1_score,roc_auc_score
from vehicle_insurance.entity.estimator import MyModel
from vehicle_insurance.entity.config_entity import ModelTrainerConfig
from imblearn.pipeline import Pipeline as ImbPipeline
import sys
import os
from pandas import DataFrame
import pandas as pd
import logging

logger=logging.getLogger('ModelTrainer')

class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig,
                 data_ingestion_artifact:DataIngestionArtifact):
        try:
            logger.info("Initializing ModelTrainer...")
            self.data_transformation_artifact = data_transformation_artifact
            self.data_ingestion_artifact=data_ingestion_artifact
            self._model_schema = read_yaml_file(MODEL_CONFIG_FILE)
            self._schema=read_yaml_file(SCHEMA_PATH)
            self.model_trainer_config = model_trainer_config
            logger.info("Model schema loaded successfully.")
        except Exception as e:
            raise MyException(e, sys) from e

    def get_model_object_and_report(self, train: DataFrame, test: DataFrame,model:ImbPipeline):
        try:
            logger.info(f"Train shape: {train.shape}, Test shape: {test.shape}")
            logger.info(f"Train columns: {train.columns.tolist()}")
            logger.info("Splitting train/test...")
            X_train, X_test, y_train, y_test = train.drop(columns=TARGET_COLUMN),test.drop(columns=TARGET_COLUMN),train[TARGET_COLUMN],test[TARGET_COLUMN]

            schema_cols = [list(d.keys())[0] for d in self._schema["columns"]]
            expected_features = [c for c in schema_cols if c not in ["_id", TARGET_COLUMN]]
            missing = [c for c in expected_features if c not in X_train.columns]
            extra = [c for c in X_train.columns if c not in schema_cols]
            if extra:
                raise Exception(f"data has some extra columns: {extra}")

            if missing:
                raise Exception(f"Training data missing schema columns: {missing}")
            

            logger.info("Fitting model...")
            model.fit(X_train, y_train)

            pipeline=MyModel(model,expected_features,schema_cols)

            logger.info("Generating predictions...")
            preds_proba=pipeline.predict_proba(X_test)
            predictions = pipeline.predict(X_test)

            logger.info("Calculating metrics...")
            recall = recall_score(y_test, predictions)
            f1 = f1_score(y_test, predictions)
            roc_score = roc_auc_score(y_test, preds_proba)
            average_precision = average_precision_score(y_test, preds_proba)

            logger.info(f"Metrics -> Recall: {recall:.4f}, F1: {f1:.4f}, ROC AUC: {roc_score:.4f}, AP: {average_precision:.4f}")

            metrics_artifact = ClassificationMetricsArtifact(
                f1=f1, roc_auc=roc_score, recall=recall, average_precision=average_precision
            )
            return pipeline,metrics_artifact, recall
        except Exception as e:
            raise MyException(e, sys) from e
        
    @staticmethod
    def read_csv(data_frame_path:str)->DataFrame:
        try:
            return pd.read_csv(data_frame_path)
        except Exception as e:
            raise MyException(e,sys) from e

    def initiate_model_trainer(self):
        try:
            logger.info("Loading train/test data...")
            train_df=self.read_csv(self.data_ingestion_artifact.train_path)
            test_df=self.read_csv(self.data_ingestion_artifact.test_path)
            logger.info("Training model and generating report...")
            logger.info("Loading preprocessing object...")
            pipeline = load_object(self.data_transformation_artifact.model_pipeline_path)
            my_model,metric_artifact, recall = self.get_model_object_and_report(train=train_df, test=test_df,model=pipeline)

            if recall < self.model_trainer_config.baseline_recall:
                logger.warning("Recall below baseline. No model will be saved.")
                raise Exception("No model found with score above the base score")

            logger.info("Saving trained model...")
            os.makedirs(os.path.dirname(self.model_trainer_config.model_save_path), exist_ok=True)
            save_object(self.model_trainer_config.model_save_path, my_model)

            logger.info("Creating ModelTrainerArtifact...")
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_path=self.model_trainer_config.model_save_path,
                metrics=metric_artifact
            )
            logger.info("Model training completed successfully.")
            return model_trainer_artifact
        except Exception as e:
            raise MyException(e, sys) from e

