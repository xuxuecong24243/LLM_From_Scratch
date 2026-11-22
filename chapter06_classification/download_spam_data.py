"""chapter06_classification: download spam data.

Original learning file: 6-1.py. Run as a module from the repository root.
"""

from project_paths import DATA_DIR
import urllib.request
import zipfile
import os
from pathlib import Path

url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
if __name__ == "__main__":
    zip_path = DATA_DIR / "downloads" / "sms_spam_collection.zip"
    extracted_path = DATA_DIR / "sms_spam_collection"
    data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"


def download_and_unzip_spam_data(
    url, zip_path, extracted_path, data_file_path
):
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download "
              "and extraction."
        )
        return

    Path(zip_path).parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response:
        with open(zip_path, "wb") as out_file:
            out_file.write(response.read())

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)

    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    print(f"File downloaded and saved as {data_file_path}")


if __name__ == "__main__":
    download_and_unzip_spam_data(
        url, zip_path, extracted_path, data_file_path
    )
