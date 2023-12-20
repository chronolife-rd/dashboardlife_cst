import streamlit as st
import boto3
import logging
import os
import streamlit as st
from botocore.exceptions import ClientError
from botocore.config import Config

def upload_file(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False

    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    s3_client = boto3.client('s3',
                             aws_secret_access_key=st.secrets.aws_credentials["AWS_SECRET_ACCESS_KEY"],
                             aws_access_key_id=st.secrets.aws_credentials["AWS_ACCESS_KEY_ID"],
                             region_name = 'eu-central-1',
                             config=Config(signature_version='s3v4'))
    try:
        s3_client.upload_file(file_name, bucket, object_name)
       
    except ClientError as e:
        logging.error(e)

def create_presigned_url(bucket_name, object_name, expiration=3600):
    """Generate a presigned URL to share an S3 object

    :param bucket_name: string
    :param object_name: string
    :param expiration: Time in seconds for the presigned URL to remain valid
    :return: Presigned URL as string. If error, returns None.
    """

    # Generate a presigned URL for the S3 object
    s3_client = boto3.client('s3',
                             aws_secret_access_key=st.secrets.aws_credentials["AWS_SECRET_ACCESS_KEY"],
                             aws_access_key_id=st.secrets.aws_credentials["AWS_ACCESS_KEY_ID"],
                             region_name = 'eu-central-1',
                             config=Config(signature_version='s3v4'))
    try:
        return  s3_client.generate_presigned_url('get_object',
                                                    Params={'Bucket': bucket_name,
                                                            'Key': object_name},
                                                    ExpiresIn=expiration)
    except ClientError as e:
        logging.error(e)
        return None

def dl_file (bucket_name, object_name,file_name):
    s3_client = boto3.client('s3',
                             aws_secret_access_key=st.secrets.aws_credentials["AWS_SECRET_ACCESS_KEY"],
                             aws_access_key_id=st.secrets.aws_credentials["AWS_ACCESS_KEY_ID"],
                             region_name = 'eu-central-1',
                             config=Config(signature_version='s3v4'))
    try :
        return s3_client.download_file(bucket_name, object_name, f'{file_name}.csv')
    except ClientError as e:
        logging.error(e)
        return e

# Send data to S3
if __name__ == "__main__":
    abd_breath = 'ecg.csv'

    bucket_name = 'chronolifedatadl'
    end_user = '3pupBp'
    object_key = f'{end_user}/{abd_breath}'

    upload_file(abd_breath,bucket_name,object_key)

    print (create_presigned_url(bucket_name, object_key))
