import logging
import os
import sys
import json
import pandas as pd
from pandas import DataFrame
from vehicle_insurance.entity.artifact_entity import DataValidationArtifact, DataIngestionArtifact
from vehicle_insurance.exception import MyException
from vehicle_insurance.utils.main_utils import read_yaml_file
from vehicle_insurance.constants import *
from vehicle_insurance.entity.config_entity import DataValidationConfig

# Configure logger

logger = logging.getLogger("DataValidation")

class DataValidation:
    def __init__(self,
                 data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_config: DataValidationConfig):
        try:
            logger.info("Initializing DataValidation...")
            self.data_ingestion_artifact = data_ingestion_artifact
            self._schema_config = read_yaml_file(SCHEMA_PATH)
            self.data_validation_config = data_validation_config
            logger.info("Schema loaded successfully from %s", SCHEMA_PATH)
        except Exception as e:
            raise MyException(e, sys) from e
        
    def validate_number_columns(self,data_frame:DataFrame)->bool:
        try:
            status = (data_frame.shape[1]==len(self._schema_config['columns']))
            logger.info('Same Number of Columns: %s',status)
            return status
        except Exception as e:
            raise MyException(e,sys) from e


    def validate_columns(self, dataframe: DataFrame) -> bool:
        try:
            errors = []

            # Extract schema column names
            schema_columns = [list(col.keys())[0] for col in self._schema_config['columns']]

            # Check missing columns
            overall_missing = [c for c in schema_columns if c not in dataframe.columns]
            numeric_missing = [c for c in self._schema_config['numerical_columns'] if c not in dataframe.columns]
            categorical_missing = [c for c in self._schema_config['categorical_columns'] if c not in dataframe.columns]

            if overall_missing or numeric_missing or categorical_missing:
                errors.append(f"Missing columns. Overall: {overall_missing}, Numeric: {numeric_missing}, Categorical: {categorical_missing}")

            # Check data types
            for col in self._schema_config['numerical_columns']:
                if col in dataframe.columns and not pd.api.types.is_numeric_dtype(dataframe[col]):
                    errors.append(f"Column {col} should be numeric but is {dataframe[col].dtype}")

            for col in self._schema_config['categorical_columns']:
                if col in dataframe.columns and not pd.api.types.is_object_dtype(dataframe[col]):
                    errors.append(f"Column {col} should be categorical but is {dataframe[col].dtype}")

            # Decide return value
            if errors:
                for err in errors:
                    logger.warning(err)
                return False
            else:
                logger.info("All schema checks passed: presence, types, categories.")
                return True

        except Exception as e:
            raise MyException(e, sys) from e



    @staticmethod
    def read_csv(file):
        try:
            logger.info("Reading CSV file: %s", file)
            return pd.read_csv(file)
        except Exception as e:
            raise MyException(e, sys) from e

    def initiate_validation(self)->DataValidationArtifact:
        try:
            logger.info("Starting validation process...")
            validation_error_msg = ""

            # Load both datasets
            datasets = {
                "train": DataValidation.read_csv(self.data_ingestion_artifact.train_path),
                "test": DataValidation.read_csv(self.data_ingestion_artifact.test_path)
            }

            for name, df in datasets.items():
                logger.info("Validating %s dataset...", name)

                # Validate number of columns
                if not self.validate_number_columns(df):
                    validation_error_msg += f"There is a mismatch in the number of columns in the {name} data. "
                else:
                    logger.info("Column count validation passed for %s data.", name)

                # Validate missing columns
                status_exist = self.validate_columns(df)
                if not status_exist:
                    validation_error_msg += f"Columns are missing in {name} dataframe. "
                else:
                    logger.info("Schema validation passed for %s data.", name)

            validation_status = len(validation_error_msg) == 0
            os.makedirs(self.data_validation_config.data_validation_dir, exist_ok=True)

            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                message=validation_error_msg,
                report_path=self.data_validation_config.report_dir
            )

            validation_report = {
                'validation_status': validation_status,
                'message': validation_error_msg
            }
            with open(self.data_validation_config.report_dir, 'w') as file:
                json.dump(validation_report, file, indent=4)

            logger.info("Validation report written to %s", self.data_validation_config.report_dir)
            logger.info("Validation process completed. Status: %s", validation_status)

            return data_validation_artifact

        except Exception as e:
            raise MyException(e, sys) from e

