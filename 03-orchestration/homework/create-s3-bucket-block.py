from time import sleep
from prefect_aws import S3Bucket, AwsCredentials
from prefect import task, flow

def set_aws_credentials():
    ## Set AWS credentials for S3 access.
    aws_credentials = AwsCredentials(
        aws_access_key_id="test",
        aws_secret_access_key="test123" # Replace with your actual secret key
    )
    aws_credentials.save("aws-credentials", overwrite=True)  # Save the credentials with a name
    
    
def create_s3_bucket(bucket_name: str):
    ## Create an S3 bucket with the given name.
    aws_cred = AwsCredentials.load("aws-credentials")  # Load the AWS credentials
    s3_bucket = S3Bucket(bucket_name=bucket_name, credentials=aws_cred)
    s3_bucket.save(bucket_name, overwrite=True)  # Save the S3 bucket block with the same name
    

if __name__ == "__main__":
    # Example usage
    set_aws_credentials()
    sleep(5) # Sleep to allow time for the saving the s3 creds
    bucket_name = "mlflow-training-bucket"  
    create_s3_bucket(bucket_name)
    


