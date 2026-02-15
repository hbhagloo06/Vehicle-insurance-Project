import boto3
from botocore.config import Config


class S3Client:
    s3_client=None
    s3_resource=None

    def __init__(self):
        
        if S3Client.s3_client is None or S3Client.s3_resource is None:

            cfg = Config(retries={"max_attempts": 10, "mode": "standard"})
            
            S3Client.s3_resource=boto3.resource('s3',config=cfg)
            S3Client.s3_client=boto3.client('s3',config=cfg)

            self.s3_client=S3Client.s3_client
            self.s3_resource=S3Client.s3_resource