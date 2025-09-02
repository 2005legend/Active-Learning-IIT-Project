from pathlib import Path
from src.utils.logger import get_logger
from src.datasets.preprocess import preprocess_raw_dataset
from src.utils.plots import save_json

# === EDIT THIS ===
RAW_DATASET_DIR = r"C:\Users\sidaa\animal_al_rl\data\dogs-vs-cats\train\train"  # <- set your raw path
# =================

PROCESSED_DIR = Path("data/processed/catsdogs_128")
LOG_FILE = Path("logs/week1_preprocess.log")

if __name__ == "__main__":
    logger = get_logger("week1", LOG_FILE)
    logger.info("Week 1: Data Prep & Baseline Setup - Preprocessing started")
    
    stats = preprocess_raw_dataset(RAW_DATASET_DIR, PROCESSED_DIR, img_size=128, val_ratio=0.2)
    save_json(stats, Path("outputs/week1/preprocess_stats.json"))
    
    logger.info(f"Preprocessing complete. Saved to: {stats['out_dir']}")