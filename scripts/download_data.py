"""Downloads the credit card default data from UCI and puts it in data/raw.
If the file is already there, it does nothing."""

import urllib.request
import zipfile
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
ZIP_PATH = RAW_DIR / "credit_default.zip"
XLS_PATH = RAW_DIR / "default of credit card clients.xls"


def download_zip():
    """Downloads the zip file from UCI, unless I already have it."""
    if ZIP_PATH.exists():
        print("Zip already here, so skipping the download:", ZIP_PATH)
        return
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading from", URL)
    with urllib.request.urlopen(URL) as response:
        ZIP_PATH.write_bytes(response.read())
    print("Saved to", ZIP_PATH)


def extract_zip():
    """Takes the files out of the zip and puts them in data/raw."""
    if XLS_PATH.exists():
        print("Excel file already here, so skipping the unzip:", XLS_PATH)
        return
    with zipfile.ZipFile(ZIP_PATH) as zf:
        names = zf.namelist()
        print("Files inside the zip:", names)
        zf.extractall(RAW_DIR)
    print("Extracted into", RAW_DIR)


if __name__ == "__main__":
    download_zip()
    extract_zip()
