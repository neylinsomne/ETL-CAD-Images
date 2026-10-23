"""Descarga y extrae los datasets en data/raw/.

- cubicasa5k   Zenodo, 5,5 GB, CC BY-NC-SA 4.0 (muro exterior anotado: "Wall External")
- cvc_fp       CVC (UAB), 26 MB, CC BY-NC 4.0 (perímetro exterior anotado: relación "outerP")
- floorplancad Google Drive, ~5 GB, CC BY-NC 4.0 (incluye la clase "curtain wall"; requiere gdown)

Reanuda descargas cortadas y salta lo que ya está extraído.
Uso: python scripts/download_data.py [--only cubicasa5k cvc_fp floorplancad]
"""

from __future__ import annotations

import argparse
import hashlib
import tarfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

from filtro_planos.datasets import RAW_DIR

CUBICASA = ("https://zenodo.org/records/2613548/files/cubicasa5k.zip?download=1", "0ce0b203d1e3c125b51087b219bd23b9")
CVC_FP = "http://dag.cvc.uab.es/DATASETS/CVC-FP.zip"
FLOORPLANCAD = {  # archivo -> (id de Google Drive, split)
    "train-00.tar.xz": ("1HcyKt6qWeXog-tRfvEjdO3O3TN91PXGL", "train"),
    "train-01.tar.xz": ("1kSS7OB_EEu7VJzb0W8DK9_nu1EvshioV", "train"),
    "test-00.tar.xz": ("1jxpYgxnLUbXEzMOsjaMPQFSuvmvHimiZ", "test"),
}


def download(url: str, dest: Path) -> None:
    """Descarga con reanudación (HTTP Range)."""
    done = dest.stat().st_size if dest.exists() else 0
    request = urllib.request.Request(url, headers={"Range": f"bytes={done}-"} if done else {})
    try:
        response = urllib.request.urlopen(request)
    except urllib.error.HTTPError as exc:
        if exc.code == 416:  # ya estaba completo
            return
        raise
    with response:
        if done and response.status != 206:  # el servidor ignoró el Range: empezar de cero
            done = 0
        total = int(response.headers.get("Content-Length", 0)) + done
        with open(dest, "ab" if done else "wb") as f, tqdm(
            total=total, initial=done, unit="B", unit_scale=True, desc=dest.name
        ) as bar:
            while chunk := response.read(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            digest.update(chunk)
    return digest.hexdigest()


def get_cubicasa() -> None:
    if (RAW_DIR / "cubicasa5k" / "train.txt").exists():
        return print("cubicasa5k: ya extraído")
    url, expected = CUBICASA
    archive = RAW_DIR / "cubicasa5k.zip"
    download(url, archive)
    if md5(archive) != expected:
        raise RuntimeError(f"MD5 de {archive} no coincide; bórralo y vuelve a correr")
    with zipfile.ZipFile(archive) as z:
        z.extractall(RAW_DIR)


def get_cvc_fp() -> None:
    if (RAW_DIR / "cvc_fp" / "ImagesGT").exists():
        return print("cvc_fp: ya extraído")
    archive = RAW_DIR / "CVC-FP.zip"
    download(CVC_FP, archive)
    with zipfile.ZipFile(archive) as z:
        z.extractall(RAW_DIR / "cvc_fp")


def get_floorplancad() -> None:
    import gdown  # extra opcional: pip install -e ".[download]"

    root = RAW_DIR / "floorplancad"
    root.mkdir(parents=True, exist_ok=True)
    for name, (drive_id, split) in FLOORPLANCAD.items():
        marker = root / f".{name}.extracted"
        if marker.exists():
            continue
        archive = root / name
        if not archive.exists():
            gdown.download(id=drive_id, output=str(archive), resume=True)
        # Los tar no traen carpeta de split y train/test repiten nombres: cada uno a su carpeta.
        with tarfile.open(archive) as tar:
            tar.extractall(root / split, filter="data")
        marker.touch()


def main() -> None:
    getters = {"cubicasa5k": get_cubicasa, "cvc_fp": get_cvc_fp, "floorplancad": get_floorplancad}
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", choices=list(getters), default=list(getters))
    args = parser.parse_args()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name in args.only:
        getters[name]()


if __name__ == "__main__":
    main()
