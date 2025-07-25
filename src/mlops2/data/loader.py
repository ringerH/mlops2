from pathlib import Path
from typing import Union
import pandas as pd

from ..config.settings import DATA_DIR, DATA_CONFIG


class DataLoader:
    """CSV reader with basic schema validation."""

    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = Path(data_dir)

    def load_data(self, file_path: Union[str, Path] | None = None) -> pd.DataFrame:
        file_path = Path(file_path) if file_path else self.data_dir / DATA_CONFIG["input_file"]
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found at '{file_path}'.")
        df = pd.read_csv(file_path)

        required = {DATA_CONFIG["text_column"], DATA_CONFIG["target_column"]}
        missing  = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")
        return df
