# Some of this code is taken from: https://github.com/pytorch/vision/blob/master/torchvision/datasets/utils.py
import hashlib
import os
import urllib
import zipfile
from typing import Callable

import tqdm


def check_md5(fpath: str, md5: str) -> bool:
    return md5 == calculate_md5(fpath)


def calculate_md5(fpath: str, chunk_size: int = 1024 * 1024) -> str:
    md5 = hashlib.md5()
    with open(fpath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


def check_integrity(fpath: str, md5: str) -> bool:
    if not os.path.isfile(fpath):
        return False
    return check_md5(fpath, md5)


def makedir_exist_ok(dir_path: str) -> None:
    os.makedirs(dir_path, exist_ok=True)


def download_and_extract_archive(
    url: str, download_dir: str, filename: str, md5: str,
) -> None:

    archive_filename = "data.zip"
    data_filename = "data.jsonl"
    download_url(url=url, download_dir=download_dir, filename=archive_filename, md5=md5)

    archive_path = os.path.join(download_dir, archive_filename)
    data_path = os.path.join(download_dir, data_filename)
    print("Extracting {} to {}".format(archive_path, data_path))
    extract_archive(from_path=archive_path, to_path=data_path)


def extract_archive(from_path: str, to_path: str) -> None:
    if to_path is None:
        to_path = os.path.dirname(from_path)

    with zipfile.ZipFile(from_path, "r") as z:
        z.extractall(to_path)

    os.remove(from_path)


def gen_bar_updater() -> Callable[[int, int, int], None]:
    pbar = tqdm(total=None)

    def bar_update(count: int, block_size: int, total_size: int) -> None:
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    return bar_update


def download_url(url: str, download_dir: str, filename: str, md5: str) -> None:
    """Download a file from a url and place it in root.
    Args:
        url (str): URL to download file from
        download_dir (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the basename of the URL
        md5 (str, optional): MD5 checksum of the download. If None, do not check
    """
    makedir_exist_ok(download_dir)
    fpath = os.path.join(download_dir, filename)

    # check if file is already present locally
    if check_integrity(fpath, md5):
        print("Using downloaded and verified file: " + fpath)
    else:  # download the file
        try:
            print("Downloading " + url + " to " + fpath)
            urllib.request.urlretrieve(url, fpath, reporthook=gen_bar_updater())  # type: ignore
        except (urllib.error.URLError, IOError) as e:  # type: ignore
            if url[:5] == "https":
                url = url.replace("https:", "http:")
                print(
                    "Failed download. Trying https -> http instead."
                    " Downloading " + url + " to " + fpath
                )
                urllib.request.urlretrieve(url, fpath, reporthook=gen_bar_updater())  # type: ignore
            else:
                raise e
        # check integrity of downloaded file
        if not check_integrity(fpath, md5):
            raise RuntimeError("File not found or corrupted.")
