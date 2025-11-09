import boto3
import os

# Your S3 bucket name
bucket_name = "randompractise1"

# Create an S3 client
s3 = boto3.client('s3')

def download_all_models(local_base_path="ml-models"):
    """
    Download all top-level model folders from the S3 bucket into the local ml-models/ directory.
    Each folder in S3 becomes a subfolder in ml-models/.
    """
    os.makedirs(local_base_path, exist_ok=True)

    # List all objects in the bucket
    paginator = s3.get_paginator('list_objects_v2')
    model_prefixes = set()

    print(f"\n🔍 Fetching model prefixes from S3 bucket: {bucket_name} ...")

    # Collect top-level prefixes (folders)
    for result in paginator.paginate(Bucket=bucket_name):
        for obj in result.get('Contents', []):
            key = obj['Key']
            if "/" in key:
                prefix = key.split("/")[0]  # Get top-level folder name
                model_prefixes.add(prefix)

    if not model_prefixes:
        print("⚠️ No folders found in S3 bucket!")
        return

    print(f"\n📦 Found model folders: {', '.join(sorted(model_prefixes))}\n")

    # Download each folder
    for model_prefix in sorted(model_prefixes):
        local_path = os.path.join(local_base_path, model_prefix)
        print(f"⬇️  Downloading model '{model_prefix}' into '{local_path}' ...")
        os.makedirs(local_path, exist_ok=True)

        paginator = s3.get_paginator('list_objects_v2')
        for result in paginator.paginate(Bucket=bucket_name, Prefix=f"{model_prefix}/"):
            for obj in result.get('Contents', []):
                s3_key = obj['Key']
                if s3_key.endswith("/"):
                    continue
                local_file = os.path.join(local_path, os.path.relpath(s3_key, model_prefix))
                os.makedirs(os.path.dirname(local_file), exist_ok=True)
                s3.download_file(bucket_name, s3_key, local_file)
                print(f"   ✅ {s3_key} -> {local_file}")

    print("\n🎉 All models downloaded successfully!")


def upload_image_to_s3(file_name, s3_prefix="ml-images", object_name=None):
    """
    Uploads an image to S3 and returns a presigned URL for temporary access.
    """
    if object_name is None:
        object_name = os.path.basename(file_name)

    object_name = f"{s3_prefix}/{object_name}"
    s3.upload_file(file_name, bucket_name, object_name)

    response = s3.generate_presigned_url(
        'get_object',
        Params={
            "Bucket": bucket_name,
            "Key": object_name
        },
        ExpiresIn=3600
    )
    return response
