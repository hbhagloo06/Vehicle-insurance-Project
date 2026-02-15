from vehicle_insurance.entity.artifact_entity import (
    ModelTrainerArtifact,
    DataIngestionArtifact,
    ModelEvaluationArtifact
)
from vehicle_insurance.utils.main_utils import load_object
from vehicle_insurance.entity.config_entity import ModelEvaluationConfig
from vehicle_insurance.entity.s3_estimator import Proj1Estimator
from vehicle_insurance.exception import MyException
from vehicle_insurance.constants import TARGET_COLUMN

from sklearn.metrics import f1_score
import pandas as pd
from typing import Optional
import sys
import logging

logger = logging.getLogger("ModelEvaluation")

class ModelEvaluation:
    def __init__(
        self,
        model_trainer_artifact: ModelTrainerArtifact,
        data_ingestion_artifact: DataIngestionArtifact,
        model_eval_config: ModelEvaluationConfig,
    ) -> None:
        logger.info("Initializing ModelEvaluation...")
        self.model_trainer_artifact = model_trainer_artifact
        self.data_ingestion_artifact = data_ingestion_artifact
        self.model_eval_config = model_eval_config
        logger.info("Initialization complete.")

    def get_best_model(self) -> Optional[Proj1Estimator]:
        """
        Retrieve the current production-grade model if it exists.
        """
        try:
            logger.info("Checking for existing production model...")
            estimator = Proj1Estimator(
                model_path=self.model_eval_config.s3_model_key_path,
                bucket_name=self.model_eval_config.s3_bucket_name,
            )
            if estimator.is_model_present():
                logger.info("Production model found.")
                return estimator
            logger.info("No production model found.")
            return None
        except Exception as e:
            logger.error("Error while fetching production model.")
            raise MyException(e, sys) from e

    def evaluate_model(self) -> ModelEvaluationArtifact:
        """
        Compare trained model against production model and decide acceptance.
        """
        try:
            logger.info("Loading trained model...")
            trained_model = load_object(self.model_trainer_artifact.trained_model_path)

            logger.info("Loading test dataset...")
            test_df = pd.read_csv(self.data_ingestion_artifact.test_path)
            X_test, y_test = test_df.drop(TARGET_COLUMN, axis=1), test_df[TARGET_COLUMN]

            logger.info("Generating predictions with trained model...")
            preds_trained_model = trained_model.predict(X_test)
            f1_trained_model = f1_score(y_test, preds_trained_model)
            logger.info(f"F1 score (trained model): {f1_trained_model:.4f}")

            logger.info("Evaluating production model (if available)...")
            production_model = self.get_best_model()

            if production_model is None:
                logger.info("No production model to evaluate.")
                acceptance = True
                f1_production_model = None
                difference = None  
            else:
                production_model_preds = production_model.predict(X_test)
                f1_production_model = f1_score(y_test, production_model_preds)
                logger.info(f"F1 score (production model): {f1_production_model:.4f}")
                difference = f1_trained_model - f1_production_model
                acceptance = difference > self.model_eval_config.changed_threshold_score
                logger.info(f"Difference in F1 scores: {difference:.4f}")
                logger.info(f"Model acceptance decision: {acceptance}")

            artifact = ModelEvaluationArtifact(
                is_model_accepted=acceptance,
                diff_f1=difference,
                s3_model_path=self.model_eval_config.s3_model_key_path,
                trained_model_path=self.model_trainer_artifact.trained_model_path,
            )
            logger.info("ModelEvaluationArtifact created successfully.")
            return artifact

        except Exception as e:
            logger.error("Error during model evaluation.")
            raise MyException(e, sys) from e


            


