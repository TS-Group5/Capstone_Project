import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
import os


def upload_image_to_public_s3(file_name, bucket_name, object_name=None, region='ap-south-1'):
    """
    Uploads an image to a public S3 bucket without requiring credentials.

    :param file_name: Path to the local file to upload.
    :param bucket_name: Name of the S3 bucket.
    :param object_name: S3 object name (default: file_name).
    :param region: AWS region where the bucket is located.
    :return: Public URL of the uploaded image or None if the upload fails.
    """
    if object_name is None:
        object_name = file_name.split("/")[-1]

    # Configure S3 client with your AWS credentials
    s3_client = boto3.client('s3',
                            region_name=region,
                            aws_access_key_id='',  # Replace with your Access Key ID
                            aws_secret_access_key='')  # Replace with your Secret Access Key

    try:
        #Upload the file
        s3_client.upload_file(file_name, bucket_name, object_name)
        print(f"File uploaded successfully to {bucket_name}/{object_name}")
    except ClientError as e:
        print(f"Error uploading file: {e}")
        return None

    # Construct the public URL
    public_url = f"https://{bucket_name}.s3.{region}.amazonaws.com/{object_name}"
    return public_url

# if __name__ == "__main__":
#     # Replace with your details
#     file_name = "/content/male3.mp4"
#     bucket_name = "aimlops-cohort3-group5-capstone-project"

#     # Upload and get the public URL
#     public_url = upload_image_to_public_s3(file_name, bucket_name)
#     if public_url:
#         print("Public URL of the uploaded file:", public_url)