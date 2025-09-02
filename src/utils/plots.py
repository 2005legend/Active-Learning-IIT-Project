from pathlib import Path
import json
import matplotlib.pyplot as plt

def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def plot_curves(curves: dict[str, list[float]], xlabel: str, ylabel: str, title: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    for label, ys in curves.items():
        xs = list(range(1, len(ys) + 1))
        plt.plot(xs, ys, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()