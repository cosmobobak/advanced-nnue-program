#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "matplotlib>=3.8.0",
#     "numpy>=1.26.0",
# ]
# ///
"""
Plot activation functions.

Usage:
    uv run scripts/plot_activation.py <activation> [--output output.png]

Activations:
    identity        f(x) = x
    relu            f(x) = max(0, x)
    crelu           f(x) = clamp(x, 0, 1)
    screlu          f(x) = clamp(x, 0, 1)^2
    swish           f(x) = x * sigmoid(x)
    hard-swish      f(x) = x * clamp(x + 3, 0, 6) / 6

Example:
    uv run scripts/plot_activation.py screlu --output screlu.png
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ACTIVATIONS = {
    "identity": lambda x: x,
    "relu": lambda x: np.maximum(0, x),
    "crelu": lambda x: np.clip(x, 0, 1),
    "screlu": lambda x: np.clip(x, 0, 1) ** 2,
    "swish": lambda x: x * (1 / (1 + np.exp(-x))),
    "hard-swish": lambda x: x * np.clip(x + 3, 0, 6) / 6,
}


def plot_activation(name: str, output_path: Path | None, minimal: bool) -> None:
    """Create and display/save activation function plot."""
    fig, ax = plt.subplots(figsize=(6, 6))

    x = np.linspace(-6, 6, 1000)
    y = ACTIVATIONS[name](x)

    ax.plot(x, y, color="black", linewidth=1.5)

    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_aspect("equal")

    ax.set_xlabel("x", fontsize=12, fontweight="bold")
    ax.set_ylabel("f(x)", fontsize=12, fontweight="bold")

    ax.set_facecolor("white")
    fig.set_facecolor("white")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_color("black")
    ax.spines["bottom"].set_linewidth(1.5)

    ax.tick_params(colors="black")
    ax.grid(False)

    if minimal:
        ax.spines["left"].set_position("center")
        ax.spines["bottom"].set_position("center")
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        plt.tick_params(
            # axis='x',          # changes apply to the x-axis
            which="both",  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            left=False,  # ticks along the bottom edge are off
            right=False,  # ticks along the top edge are off
            labelbottom=False,
            labelleft=False,
        )  # labels along the bottom edge are off

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {output_path}")
    else:
        plt.show()


def main() -> None:
    """Main entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    activation = sys.argv[1]
    output_path = None

    minimal = False

    i = 2
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--output" and i + 1 < len(sys.argv):
            output_path = Path(sys.argv[i + 1])
            i += 2
        elif arg == "--minimal" or arg == "-m":
            minimal = True
            i += 1
        else:
            print(f"Error: Unknown argument '{arg}'", file=sys.stderr)
            sys.exit(1)

    if activation not in ACTIVATIONS:
        print(f"Error: Unknown activation '{activation}'", file=sys.stderr)
        print(f"Available: {', '.join(ACTIVATIONS.keys())}", file=sys.stderr)
        sys.exit(1)

    plot_activation(activation, output_path, minimal)


if __name__ == "__main__":
    main()
