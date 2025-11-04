"""
S3 helper functions for uploading images to AWS S3.
"""

import os
import time
import logging
from typing import Optional
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


def get_s3_client():
    """
    Get S3 client if credentials are configured.
    Returns None if S3 is not configured.
    """
    try:
        import boto3

        # Check if S3 is enabled
        if os.getenv("CLOUD_STORAGE_ENABLED", "false").lower() != "true":
            return None

        # Check for required credentials
        access_key = os.getenv("AWS_ACCESS_KEY_ID")
        secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        region = os.getenv("AWS_REGION", "us-east-1")

        if not access_key or not secret_key:
            logger.warning(
                "S3 credentials not found. Image upload to S3 will be disabled."
            )
            return None

        return boto3.client(
            "s3",
            region_name=region,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )
    except ImportError:
        logger.warning("boto3 not installed. Image upload to S3 will be disabled.")
        return None
    except Exception as e:
        logger.error(f"Error initializing S3 client: {e}")
        return None


def upload_image_to_s3(
    image_data: bytes,
    image_key: str,
    content_type: str = "image/jpeg",
    bucket_name: Optional[str] = None,
) -> Optional[str]:
    """
    Upload image data to S3 and return a public URL (non-expiring).

    Args:
        image_data: Image data as bytes
        image_key: S3 key/path for the image (e.g., "images/img_xxx.jpg")
        content_type: MIME type of the image
        bucket_name: S3 bucket name (defaults to S3_BUCKET_NAME env var)

    Returns:
        Public URL if successful, None otherwise
    """
    try:
        s3_client = get_s3_client()
        if not s3_client:
            return None

        if not bucket_name:
            bucket_name = os.getenv("S3_BUCKET_NAME", "document-extractor-files")

        region = os.getenv("AWS_REGION", "us-east-1")

        # Upload to S3 with public-read ACL
        try:
            s3_client.put_object(
                Bucket=bucket_name,
                Key=image_key,
                Body=image_data,
                ContentType=content_type,
                ACL="public-read",
                Metadata={
                    "uploaded-at": str(int(time.time())),
                    "upload-type": "paddleocr-image",
                },
            )
        except ClientError as acl_error:
            # If ACL fails (bucket policy might block it), try without ACL
            # The bucket policy might still allow public access
            logger.warning(
                f"⚠️ Could not set ACL='public-read' for {image_key}: {acl_error}. Uploading without ACL."
            )
            s3_client.put_object(
                Bucket=bucket_name,
                Key=image_key,
                Body=image_data,
                ContentType=content_type,
                Metadata={
                    "uploaded-at": str(int(time.time())),
                    "upload-type": "paddleocr-image",
                },
            )

        logger.info(f"✅ Image uploaded to S3: {image_key}")

        # Generate public URL (non-expiring)
        # Format: https://{bucket}.s3.{region}.amazonaws.com/{key}
        # For us-east-1, the format is: https://{bucket}.s3.amazonaws.com/{key}
        if region == "us-east-1":
            public_url = f"https://{bucket_name}.s3.amazonaws.com/{image_key}"
        else:
            public_url = f"https://{bucket_name}.s3.{region}.amazonaws.com/{image_key}"

        return public_url

    except ClientError as e:
        logger.error(f"❌ S3 upload error for {image_key}: {e}")
        return None
    except Exception as e:
        logger.error(f"❌ Error uploading image to S3: {e}")
        return None


def get_content_type_from_filename(filename: str) -> str:
    """
    Get MIME type from filename extension.

    Args:
        filename: Image filename

    Returns:
        MIME type string
    """
    ext = filename.lower().split(".")[-1] if "." in filename else ""
    content_types = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "gif": "image/gif",
        "webp": "image/webp",
        "bmp": "image/bmp",
        "svg": "image/svg+xml",
    }
    return content_types.get(ext, "image/jpeg")
