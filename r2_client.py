"""
Cloudflare R2 client (S3-compatible) for the meet-recorder pipeline.

Handles upload, download, pre-signed URL generation, and deletion of audio
and transcript artifacts. Uses boto3 with the R2 endpoint URL.

Environment variables expected:
    R2_ACCESS_KEY_ID
    R2_SECRET_ACCESS_KEY
    R2_BUCKET          (defaults to "meet-recorder")
    R2_ENDPOINT_URL    (e.g. https://<account-id>.r2.cloudflarestorage.com)
    R2_REGION          (defaults to "auto" which R2 accepts)
"""

import os
import logging
from pathlib import Path
from typing import Optional

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

logger = logging.getLogger(__name__)


DEFAULT_BUCKET = "meet-recorder"
DEFAULT_REGION = "auto"


class R2ClientError(RuntimeError):
    """Raised when R2 operations fail in a way callers should handle."""


def _build_client():
    access_key = os.getenv("R2_ACCESS_KEY_ID")
    secret_key = os.getenv("R2_SECRET_ACCESS_KEY")
    endpoint_url = os.getenv("R2_ENDPOINT_URL")
    region = os.getenv("R2_REGION", DEFAULT_REGION)

    if not access_key or not secret_key:
        raise R2ClientError(
            "R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY must be set in env"
        )
    if not endpoint_url:
        raise R2ClientError("R2_ENDPOINT_URL must be set in env")

    # R2 wants path-style addressing and SigV4
    config = Config(
        signature_version="s3v4",
        s3={"addressing_style": "path"},
        retries={"max_attempts": 3, "mode": "standard"},
    )

    return boto3.client(
        "s3",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        endpoint_url=endpoint_url,
        region_name=region,
        config=config,
    )


def _get_bucket() -> str:
    return os.getenv("R2_BUCKET", DEFAULT_BUCKET)


def upload_file(local_path: str, key: str, content_type: Optional[str] = None) -> str:
    """
    Upload a file from disk to R2.

    Returns the object key. Raises R2ClientError on failure.
    """
    if not os.path.exists(local_path):
        raise R2ClientError(f"Local file not found: {local_path}")

    client = _build_client()
    bucket = _get_bucket()
    extra = {"ContentType": content_type} if content_type else {}

    try:
        size = os.path.getsize(local_path)
        logger.info(f"R2 upload start: {local_path} -> {bucket}/{key} ({size/1024/1024:.1f}MB)")
        client.upload_file(local_path, bucket, key, ExtraArgs=extra)
        logger.info(f"R2 upload done: {bucket}/{key}")
        return key
    except (BotoCoreError, ClientError) as e:
        raise R2ClientError(f"R2 upload failed for {key}: {e}") from e


def upload_bytes(data: bytes, key: str, content_type: str = "application/octet-stream") -> str:
    """Upload an in-memory bytes blob to R2."""
    client = _build_client()
    bucket = _get_bucket()
    try:
        logger.info(f"R2 upload bytes: {bucket}/{key} ({len(data)/1024/1024:.2f}MB)")
        client.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)
        return key
    except (BotoCoreError, ClientError) as e:
        raise R2ClientError(f"R2 upload_bytes failed for {key}: {e}") from e


def upload_text(text: str, key: str, content_type: str = "text/plain; charset=utf-8") -> str:
    """Upload a UTF-8 string to R2."""
    return upload_bytes(text.encode("utf-8"), key=key, content_type=content_type)


def download_file(key: str, local_path: str) -> str:
    """
    Download an R2 object to a local path.

    Returns the local path. Raises R2ClientError on failure.
    """
    client = _build_client()
    bucket = _get_bucket()
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        logger.info(f"R2 download start: {bucket}/{key} -> {local_path}")
        client.download_file(bucket, key, local_path)
        size = os.path.getsize(local_path)
        logger.info(f"R2 download done: {local_path} ({size/1024/1024:.1f}MB)")
        return local_path
    except (BotoCoreError, ClientError) as e:
        raise R2ClientError(f"R2 download failed for {key}: {e}") from e


def generate_presigned_url(key: str, expires_in: int = 3600) -> str:
    """
    Generate a pre-signed GET URL for an R2 object.

    expires_in is in seconds (default 1 hour).
    """
    client = _build_client()
    bucket = _get_bucket()
    try:
        url = client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=expires_in,
        )
        return url
    except (BotoCoreError, ClientError) as e:
        raise R2ClientError(f"R2 presign failed for {key}: {e}") from e


def delete_object(key: str) -> None:
    """Delete an R2 object. Safe to call on missing keys."""
    client = _build_client()
    bucket = _get_bucket()
    try:
        client.delete_object(Bucket=bucket, Key=key)
        logger.info(f"R2 deleted: {bucket}/{key}")
    except (BotoCoreError, ClientError) as e:
        # Non-fatal: log and continue
        logger.warning(f"R2 delete failed for {key}: {e}")


def object_exists(key: str) -> bool:
    """Check whether an R2 object exists."""
    client = _build_client()
    bucket = _get_bucket()
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("404", "NoSuchKey", "NotFound"):
            return False
        raise R2ClientError(f"R2 head_object failed for {key}: {e}") from e
