from pathlib import Path
import logging
import sys

def get_logger(name: str, log_file: Path, level=logging.INFO) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # avoid duplicate logs
    
    # Clear existing handlers if re-created in notebooks
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)
            h.close()
    
    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # File handler (UTF-8 safe on Windows)
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    
    # Console handler (no emojis to avoid cp1252 issues)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    
    logger.info(f"Logger initialized. Writing logs to {log_file}")
    return logger