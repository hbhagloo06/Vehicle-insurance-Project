from vehicle_insurance.logger import configure_logging
from vehicle_insurance.exception import MyException
import sys
from vehicle_insurance.pipline.training_pipeline import TrainPipeline
import logging

configure_logging()   # ← This activates logging globally

logger = logging.getLogger(__name__)
logger.info("Main started")


""" try:
    a=1+'Z'
except Exception as e:
    
    logger.info(e)
    raise MyException(e,sys) from e """

train=TrainPipeline()
train.run_pipeline()



