from dotenv import load_dotenv
import os
load_dotenv(r'C:\Users\Asus\Downloads\Vehicle-insurance-Project\.env')

DATABASE_NAME='vehicle_db'
COLLECTION='my_collection'
MONGO_DB_URL=os.getenv('MONGO_URL')

ARTIFACT_DIR:str='artifacts'
PIPELINE:str=''

TEST_SIZE=0.25

FEATURE_STORE_DIR='feature_store'
INGESTED_DIR='ingested'

train_file='train.csv'
test_file='test.csv'
raw_file='raw.csv'

SCHEMA_PATH='config/schema.yaml'

DATA_VALIDATION_DIR='data_validation'
DATA_VALIDATION_REPORT='validation_report.yaml'


#TRANSFORMATION
DATA_TRANSFORMED_DIR='transformed'
train_transformed='train_transformed.csv'
test_transformed='test_transformed.csv'
preprocess_object='preprocess.dill'

TARGET_COLUMN='Response'

#model trainer
MODEL_CONFIG_FILE='config/model.yaml'
MODEL_SAVE_FILE='model.pkl'
MODEL_DIR='model'
MODEL_TRAINER_EXPECTED_RECALL=0.35



AWS_ACCESS_KEY_ID=os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY=os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION=os.getenv('AWS_REGION','us-east-1')


S3_BUCKET_NAME='vehicle-insurance-bucket11'
BUCKET_S3_KEY='model_registry'
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE:float=0.02

APP_HOST='0.0.0.0'
PORT=8000
