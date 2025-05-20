import boto3
import os

def parse_s3_uri(s3_uri):
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
    parts = s3_uri.replace("s3://", "").split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid S3 URI structure: {s3_uri}")
    return parts[0], parts[1]

def download_single_image_from_s3(s3_uri, local_dir):
    access_key = "AKIA3TVWE5ELUK6JG2WK"
    secret_key = "jTL7V8vEq0r5dFR5VNCRkWL66v3yg9t55bDJlLTw"  # Replace with your actual secret key

    os.makedirs(local_dir, exist_ok=True)
    bucket, key = parse_s3_uri(s3_uri)
    filename = os.path.basename(key)
    local_path = os.path.join(local_dir, filename)

    s3 = boto3.client(
        's3',
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key
    )
    s3.download_file(bucket, key, local_path)
    return [local_path]

