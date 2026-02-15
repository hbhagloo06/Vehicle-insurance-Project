import sys
from vehicle_insurance.entity.config_entity import VehiclePredictorConfig
from vehicle_insurance.entity.s3_estimator import Proj1Estimator
from vehicle_insurance.exception import MyException
import logging
from pandas import DataFrame
import numpy as np

logger=logging.getLogger('Production Pipeline')
class VehicleData:
    def __init__(
        self,
        Gender: str,
        Age: int,
        Driving_License: int,
        Region_Code: int,
        Previously_Insured: int,
        Annual_Premium: float,
        Policy_Sales_Channel: int,
        Vintage: int,
        Vehicle_Age: str,
        Vehicle_Damage: str
    ):
        """
        Vehicle Data constructor
        Input: all features of the trained model for prediction
        """
        try:
            self.Gender = Gender
            self.Age = Age
            self.Driving_License = Driving_License
            self.Region_Code = Region_Code
            self.Previously_Insured = Previously_Insured
            self.Annual_Premium = Annual_Premium
            self.Policy_Sales_Channel = Policy_Sales_Channel
            self.Vintage = Vintage
            self.Vehicle_Age = Vehicle_Age
            self.Vehicle_Damage = Vehicle_Damage

        except Exception as e:
            raise MyException(e, sys) from e

    def get_vehicle_input_data_frame(self)-> DataFrame:
        """
        This function returns a DataFrame from input
        """
        try:
            
            vehicle_input_dict = self.get_vehicle_data_as_dict()
            return DataFrame(vehicle_input_dict)
        
        except Exception as e:
            raise MyException(e, sys) from e


    def get_vehicle_data_as_dict(self):
        """
        This function returns a dictionary from VehicleData class input
        """
        logger.info("Entered get_vehicle_data_as_dict method as VehicleData class")

        try:
            input_data = {
                "Gender": [self.Gender],
                "Age": [self.Age],
                "Driving_License": [self.Driving_License],
                "Region_Code": [self.Region_Code],
                "Previously_Insured": [self.Previously_Insured],
                "Annual_Premium": [self.Annual_Premium],
                "Policy_Sales_Channel": [self.Policy_Sales_Channel],
                "Vintage": [self.Vintage],
                "Vehicle_Age": [self.Vehicle_Age],
                "Vehicle_Damage": [self.Vehicle_Damage]
            }

            logger.info("Created vehicle data dict")
            logger.info("Exited get_vehicle_data_as_dict method as VehicleData class")
            return input_data

        except Exception as e:
            raise MyException(e, sys) from e