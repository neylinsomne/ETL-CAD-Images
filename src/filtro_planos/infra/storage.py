"""Archivos (originales y salidas): disco local o un bucket S3 (o compatible, p. ej. MinIO).

Claves usadas:
    originales/<plano_id>/<nombre>
    procesado/<plano_id>/p<n>/<salida>.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from filtro_planos.config import Settings


class Storage(Protocol):
    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str: ...
    def get(self, key: str) -> bytes: ...
    def exists(self, key: str) -> bool: ...
    def ping(self) -> bool: ...


class LocalStorage:
    def __init__(self, root: Path):
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        path = (self.root / key).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError(f"Clave fuera del almacenamiento: {key}")
        return path

    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return key

    def get(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def ping(self) -> bool:
        self.root.mkdir(parents=True, exist_ok=True)
        return True


class S3Storage:
    def __init__(self, bucket: str, endpoint_url: str | None = None, client=None):
        import boto3

        self.bucket = bucket
        self.client = client or boto3.client("s3", endpoint_url=endpoint_url)

    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType=content_type)
        return key

    def get(self, key: str) -> bytes:
        return self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as exc:
            if exc.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def ping(self) -> bool:
        self.client.head_bucket(Bucket=self.bucket)
        return True


def build_storage(settings: Settings) -> Storage:
    if settings.storage_backend == "local":
        return LocalStorage(settings.storage_root)
    if settings.storage_backend == "s3":
        return S3Storage(settings.s3_bucket, settings.s3_endpoint_url)
    raise ValueError(f"STORAGE_BACKEND desconocido: {settings.storage_backend!r} (local | s3)")
