"""项目资源位置；导入本模块不会创建目录或下载数据。"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "gpt2"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
