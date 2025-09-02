from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
import shutil

def _infer_label_from_name(p: Path) -> str | None:
    # Check filename first
    n = p.name.lower()
    if "cat" in n:
        return "cat"
    if "dog" in n:
        return "dog"
    
    # Check parent folder name
    parent_name = p.parent.name.lower()
    if "cat" in parent_name:
        return "cat"
    if "dog" in parent_name:
        return "dog"
    
    return None

def preprocess_raw_dataset(
    raw_dir: str | Path,
    out_dir: str | Path,
    img_size: int = 128,  # Changed to match your spec (128x128)
    val_ratio: float = 0.2,
) -> dict:
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    train_dir = out_dir / "train"
    val_dir = out_dir / "val"
    
    # Clean & make dirs
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (train_dir / "cat").mkdir(parents=True, exist_ok=True)
    (train_dir / "dog").mkdir(parents=True, exist_ok=True)
    (val_dir / "cat").mkdir(parents=True, exist_ok=True)
    (val_dir / "dog").mkdir(parents=True, exist_ok=True)
    
    # Collect files - check both root and subdirectories
    all_imgs = []
    
    # First check root directory
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
        all_imgs.extend(list(raw_dir.glob(ext)))
    
    # Then check all subdirectories recursively
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
        all_imgs.extend(list(raw_dir.rglob(ext)))
    
    # Filter valid labelable images
    pairs = []
    for p in all_imgs:
        lbl = _infer_label_from_name(p)
        if lbl is not None:
            pairs.append((p, lbl))
    
    paths, labels = zip(*pairs) if pairs else ([], [])
    if not paths:
        raise RuntimeError(f"No labelable images found in {raw_dir}. Ensure filenames contain 'cat' or 'dog'.")
    
    tr_paths, va_paths, tr_labels, va_labels = train_test_split(
        paths, labels, test_size=val_ratio, random_state=42, stratify=labels
    )
    
    # Resize & save
    def _save_batch(paths, labels, root):
        for src, lbl in zip(paths, labels):
            try:
                img = Image.open(src).convert("RGB").resize((img_size, img_size))
                dst = Path(root) / lbl / src.name
                img.save(dst)
            except Exception:
                # Skip unreadable images silently; training will be robust
                pass
    
    _save_batch(tr_paths, tr_labels, train_dir)
    _save_batch(va_paths, va_labels, val_dir)
    
    return {
        "train_count": len(tr_paths),
        "val_count": len(va_paths),
        "out_dir": str(out_dir),
        "img_size": img_size,
    }