from vehicle_insurance.configuration.mongo_db_connection import MongoDB
from vehicle_insurance.constants import DATABASE_NAME
import pandas as pd
import numpy as np
import sys
from typing import Optional
from vehicle_insurance.exception import MyException

class VehicleData:
    def __init__(self):
        try:
            self.mongo_client=MongoDB(db_name=DATABASE_NAME)
        except Exception as e:
            raise MyException(e,sys) from e
        
    def export_collection_as_dataframe(self,collection_name:str,database_name:Optional[str]=None):
        try:
            if database_name is None:
                collection=self.mongo_client.database[collection_name]
            else:
                collection=self.mongo_client[database_name][collection_name]

            print('Fetching data from mongo')
            data=pd.DataFrame(list(collection.find()))
            print(f'Fetched data with rows:{data.shape[0]}, columns:{data.shape[1]}')
            if 'id' in data:
                data.drop(columns=['id'],inplace=True)
            data.replace({'na':np.nan},inplace=True)

            return data
        except Exception as e:
            raise MyException(e,sys) from e