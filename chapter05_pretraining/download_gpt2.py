"""chapter05_pretraining: download gpt2.

Original learning file: 5.5.py. Run as a module from the repository root.
"""

from project_paths import MODELS_DIR
from gpt_download import download_and_load_gpt2

if __name__ == "__main__":
    settings, params = download_and_load_gpt2(
        model_size="124M",models_dir=MODELS_DIR
    )
