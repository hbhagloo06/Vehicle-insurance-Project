from vehicle_insurance.entity.config_entity import DataIngestionConfig
import sys
import logging
from vehicle_insurance.data_access.proj1_data import VehicleData
from vehicle_insurance.exception import MyException
from sklearn.model_selection import train_test_split
from pandas import DataFrame
import os
from vehicle_insurance.entity.artifact_entity import DataIngestionArtifact

logger = logging.getLogger(__name__)

class DataIngestion:

    def __init__(self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        try:
            self.data_ingestion_config = data_ingestion_config
            logger.info("Initialized DataIngestion with config: %s", data_ingestion_config)
        except Exception as e:
            raise MyException(e, sys) from e

    def export_data(self) -> DataFrame:
        try:
            logger.info("Exporting data from MongoDB collection: %s", self.data_ingestion_config.collection)
            mydata = VehicleData()
            dataframe = mydata.export_collection_as_dataframe(self.data_ingestion_config.collection)

            feature_store_dir = self.data_ingestion_config.feature_store_dir
            os.makedirs(feature_store_dir, exist_ok=True)
            dataframe.to_csv(self.data_ingestion_config.feature_store_file, index=False, header=True)
            logger.info("Data exported and saved to feature store at: %s", feature_store_dir)

            return dataframe

        except Exception as e:
            logger.error("Error occurred during data export: %s", e)
            raise MyException(e, sys) from e

    def train_test_split(self, dataframe: DataFrame) -> None:
        try:
            logger.info("Splitting data into train and test sets with test_size=%s",
                        self.data_ingestion_config.test_split)
            train_df, test_df = train_test_split(
                dataframe,
                test_size=self.data_ingestion_config.test_split,
                random_state=42
            )
            os.makedirs(self.data_ingestion_config.ingested_dir,exist_ok=True)
            train_path = self.data_ingestion_config.train_path
            test_path = self.data_ingestion_config.test_path

            train_df.to_csv(train_path, index=False, header=True)
            test_df.to_csv(test_path, index=False, header=True)

            logger.info("Train set saved to: %s", train_path)
            logger.info("Test set saved to: %s", test_path)

        except Exception as e:
            logger.error("Error occurred during train-test split: %s", e)
            raise MyException(e, sys) from e

    def initiate_data_ingestion(self):
        try:
            logger.info("Starting data ingestion process...")
            dataframe = self.export_data()
            self.train_test_split(dataframe)

            dataingestion_artifact = DataIngestionArtifact(
                train_path=self.data_ingestion_config.train_path,
                test_path=self.data_ingestion_config.test_path
            )
            logger.info("Data ingestion completed successfully. Artifact created.")

            return dataingestion_artifact

        except Exception as e:
            logger.error("Error occurred during data ingestion: %s", e)
            raise MyException(e, sys) from e


            
