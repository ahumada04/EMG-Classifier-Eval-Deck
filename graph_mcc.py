import os
import argparse
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")          
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_scalars(logdir: str, tag_filter: str = "MCC/Test"):
    results = {}

    for root, dirs, files in os.walk(logdir):
        event_files = [f for f in files if f.startswith("events.out.tfevents")]
        if not event_files:
            continue

        ea = EventAccumulator(root, size_guidance={"scalars": 0})
        try:
            ea.Reload()
        except Exception as e:
            print(f"  [WARN] could not load {root}: {e}")
            continue

        for tag in ea.Tags().get("scalars", []):
            if tag_filter not in tag:
                continue

            events = ea.Scalars(tag)
            if not events:
                continue

            # Take the last recorded value (final epoch / step)
            last_value = events[-1].value

            # Build a short, readable label from the tag
            # e.g.  "MCC/Test/MW256_S32_LR0.001_E20"  →  "MW256_S32_LR0.001_E20"
            label = tag.split("/")[-1] if "/" in tag else tag

            # If we see the same label from multiple runs, keep the max
            if label in results:
                results[label] = max(results[label], last_value)
            else:
                results[label] = last_value

    return results


def shorten_label(label: str, max_len: int = 24) -> str:
    """Trim label so x-axis ticks stay readable."""
    return label if len(label) <= max_len else label[:max_len - 1] + "…"


def plot_bar_chart(data: dict, title: str = "MCC — Test Scalars", outfile: str = "tb_bar_chart.png"):
    if not data:
        print("No matching scalars found. Check --logdir and --tag.")
        return

    # Sort by value descending so the best model is on the left
    labels, values = zip(*sorted(data.items(), key=lambda x: x[1], reverse=True))
    short_labels = [shorten_label(l) for l in labels]

    n = len(labels)
    fig_width = max(10, n * 0.9)          # grow with number of bars
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    # Color gradient (blue --> teal)
    colors = plt.cm.Blues_r([0.3 + 0.5 * i / max(n - 1, 1) for i in range(n)])

    bars = ax.bar(range(n), values, color=colors, edgecolor="white", linewidth=0.8, zorder=3)

    # Value labels on top of each bar
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.004,
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#333"
        )

    ax.set_xticks(range(n))
    ax.set_xticklabels(short_labels, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("MCC Score", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
    ax.set_xlim(-0.6, n - 0.4)

    # Sensible y-axis range
    ymin = max(0, min(values) - 0.05)
    ymax = min(1.0, max(values) + 0.08)
    ax.set_ylim(ymin, ymax)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))

    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"Saved → {outfile}")

    # Also try to show interactively (works in notebooks / local environments)
    try:
        matplotlib.use("TkAgg")
        plt.show()
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Bar chart of TensorBoard scalars")
    parser.add_argument("--logdir", default="./logs",
                        help="Root directory that contains your TensorBoard run folders")
    parser.add_argument("--tag", default="MCC/Test",
                        help="Substring filter for scalar tags (default: 'MCC/Test')")
    parser.add_argument("--title", default="MCC — Test Scalars",
                        help="Chart title")
    parser.add_argument("--out", default="tb_bar_chart.png",
                        help="Output image filename")
    args = parser.parse_args()

    print(f"Scanning: {os.path.abspath(args.logdir)}")
    print(f"Tag filter: '{args.tag}'")

    data = load_scalars(args.logdir, tag_filter=args.tag)

    if data:
        print(f"\nFound {len(data)} scalars:")
        for k, v in sorted(data.items(), key=lambda x: x[1], reverse=True):
            print(f"  {k:45s}  {v:.6f}")
    else:
        print("No data found.")

    plot_bar_chart(data, title=args.title, outfile=args.out)


if __name__ == "__main__":
    main()