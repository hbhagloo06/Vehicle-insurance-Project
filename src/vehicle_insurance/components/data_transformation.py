from vehicle_insurance.constants import *
from vehicle_insurance.entity.artifact_entity import (
    DataValidationArtifact,
    DataIngestionArtifact,
    DataTransformationArtifact
)
from vehicle_insurance.utils.main_utils import (
    read_yaml_file,
    save_object
)
from vehicle_insurance.entity.config_entity import DataTranformationConfig
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
import pandas as pd
from pandas import DataFrame
import sys
from sklearn.ensemble import RandomForestClassifier
from vehicle_insurance.exception import MyException
import logging

logger = logging.getLogger("DataTransformation")


class DataTransformation:
    def __init__(
        self,
        data_validation_artifact: DataValidationArtifact,
        data_ingestion_artifact: DataIngestionArtifact,
        data_transformation_config: DataTranformationConfig
    ) -> None:
        logger.info("Initializing DataTransformation class...")
        self.data_validation_artifact = data_validation_artifact
        self.data_ingestion_artifact = data_ingestion_artifact
        self._schema_config = read_yaml_file(SCHEMA_PATH)
        self._model_config=read_yaml_file(MODEL_CONFIG_FILE)
        self.data_transformation_config = data_transformation_config
        logger.info("Schema config loaded successfully.")

    @staticmethod
    def read_csv(df_path: str) -> DataFrame:
        logger.info(f"Reading CSV file from path: {df_path}")
        try:
            df = pd.read_csv(df_path)
            logger.info(f"CSV file loaded with shape {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error reading CSV at {df_path}: {e}")
            raise MyException(e, sys) from e

    def get_data_transformer(self) -> ColumnTransformer:
        logger.info("Creating preprocessing pipeline...")
        one_hot_cols = self._schema_config["onehot_columns"]

        preprocesser = ColumnTransformer(
            transformers=[
                ("one_hot", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), one_hot_cols),
            ],
            remainder="passthrough"
        )
        logger.info("Preprocessing Transformer created successfully.")
        return preprocesser

    def _remove_columns(self, df: DataFrame) -> DataFrame:
        drop_columns = self._schema_config["drop_columns"]
        df_out = df.drop(columns=[col for col in drop_columns if col in df.columns])
        logger.info(f"Columns after removal: {df_out.columns.tolist()}")
        return df_out

    def _map_vehicle_age(self, df: DataFrame) -> DataFrame:
        logger.info("Mapping Vehicle_Age column...")
        df["Vehicle_Age"] = df["Vehicle_Age"].map({
            "< 1 Year": 0,
            "1-2 Year": 1,
            "> 2 Years": 2
        })
        logger.info("Vehicle_Age mapping complete.")
        return df

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        logger.info("Starting data transformation process...")
        if not self.data_validation_artifact.validation_status:
            logger.error("Data validation failed.")
            raise Exception(self.data_validation_artifact.message)

        try:
            preprocessor_transformer = self.get_data_transformer()
            params=self._model_config['model_parameters']
            pipeline = ImbPipeline(steps=[
                ('map_vehicle_age', FunctionTransformer(self._map_vehicle_age)),
                ('remove_cols', FunctionTransformer(self._remove_columns)),
                ('encoder', preprocessor_transformer),
                ('sampler', SMOTEENN(sampling_strategy='minority', random_state=42)),
                ('model', RandomForestClassifier(**params))
            ])

            # Save preprocessor object
            save_object(self.data_transformation_config.model_pipeline_path, pipeline)
            logger.info("Pipeline saved successfully.")

            return DataTransformationArtifact(
                model_pipeline_path=self.data_transformation_config.model_pipeline_path
            )

        except Exception as e:
            logger.error(f"Error during data transformation: {e}")
            raise MyException(e, sys) from e








        

    



    
