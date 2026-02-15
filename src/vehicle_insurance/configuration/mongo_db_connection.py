
from vehicle_insurance.constants import DATABASE_NAME,COLLECTION,MONGO_DB_URL
import pymongo
import sys
import logging
import certifi

from pymongo import MongoClient
from vehicle_insurance.logger import configure_logging
from vehicle_insurance.exception import MyException
configure_logging()

logger=logging.getLogger(__name__)
ca=certifi.where()
class MongoDB:
    client=None
    def __init__(self,db_name:str=DATABASE_NAME):
     try:
            if MongoDB.client is None:
                mongo_db_url=MONGO_DB_URL
                if mongo_db_url is None:
                    raise Exception('Error in retrieving the mongo db url')
                
                MongoDB.client=MongoClient(mongo_db_url,tlsCAFile=ca)

            self.client=MongoDB.client
            self.database=self.client[db_name]
            self.database_name=db_name
            logging.info('MongoDB connection was succesfull')

     except Exception as e:
         raise MyException(e,sys) from e
