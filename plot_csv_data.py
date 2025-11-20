#!/usr/bin/env -S uv run
"""
Script to plot CSV data from chess analysis files.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from typing import List, Tuple


def setup_plot_style():
    """Set up matplotlib style for better-looking plots."""
    plt.style.use(
        "seaborn-v0_8" if "seaborn-v0_8" in plt.style.available else "default"
    )
    plt.rcParams["figure.figsize"] = (12, 8)
    plt.rcParams["font.size"] = 10
    plt.rcParams["lines.linewidth"] = 2


def plot_length_distribution(csv_file: str = "length_counts.csv") -> None:
    """Plot distribution of game lengths."""
    try:
        df = pd.read_csv(csv_file)

        plt.figure(figsize=(14, 8))
        plt.plot(df["length"], df["count"], color="blue", alpha=0.7)
        plt.fill_between(df["length"], df["count"], alpha=0.3, color="blue")

        plt.xlabel("Game Length (moves)")
        plt.ylabel("Count")
        plt.title("Distribution of Chess Game Lengths")
        plt.grid(True, alpha=0.3)

        # Add statistics
        total_games = df["count"].sum()
        weighted_mean = np.average(df["length"], weights=df["count"])

        plt.text(
            0.7,
            0.9,
            f"Total games: {total_games:,}\nMean length: {weighted_mean:.1f} moves",
            transform=plt.gca().transAxes,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        plt.tight_layout()
        plt.savefig("game_length_distribution.png", dpi=300, bbox_inches="tight")
        plt.show()

    except FileNotFoundError:
        print(f"File {csv_file} not found")
    except Exception as e:
        print(f"Error plotting length distribution: {e}")


def plot_evaluation_distribution(csv_file: str = "eval_counts.csv") -> None:
    """Plot distribution of position evaluations."""
    try:
        df = pd.read_csv(csv_file)

        # Convert eval to numeric, handling potential string prefixes
        if df["eval"].dtype == "object":
            df["eval"] = pd.to_numeric(df["eval"], errors="coerce")

        # Remove any NaN values
        df = df.dropna()

        # Take absolute values of all evals for plotting
        df["eval"] = df["eval"].abs()

        # remove evals with value < 5
        df = df[df["eval"] >= 5]

        plt.figure(figsize=(14, 8))

        # Create histogram-like plot
        plt.bar(
            df["eval"],
            df["count"],
            width=1,
            alpha=0.7,
            color="red",
            edgecolor="darkred",
            linewidth=0.1,
        )

        plt.xlabel("Position Evaluation (centipawns)")
        plt.ylabel("Count")
        plt.title("Distribution of Chess Position Evaluations")
        plt.grid(True, alpha=0.3)

        # Add vertical line at 0 (equal position)
        plt.axvline(
            x=0, color="black", linestyle="--", alpha=0.7, label="Equal position"
        )
        plt.legend()

        # Limit x-axis to reasonable range for visibility
        # eval_range = df["eval"].max() - df["eval"].min()
        # if eval_range > 2000:
        plt.xlim(-1, 5_000)

        # x-axis logarithmic scale if needed

        plt.tight_layout()
        plt.savefig("evaluation_distribution.png", dpi=300, bbox_inches="tight")
        plt.show()

    except FileNotFoundError:
        print(f"File {csv_file} not found")
    except Exception as e:
        print(f"Error plotting evaluation distribution: {e}")


def plot_material_distribution(csv_file: str = "material_counts.csv") -> None:
    """Plot distribution of material counts."""
    try:
        df = pd.read_csv(csv_file)

        plt.figure(figsize=(14, 8))
        plt.plot(
            df["men"], df["count"], marker="o", markersize=4, color="green", alpha=0.8
        )
        plt.fill_between(df["men"], df["count"], alpha=0.3, color="green")

        plt.xlabel("Number of Pieces on Board")
        plt.ylabel("Count")
        plt.title("Distribution of Piece Counts in Chess Positions")
        plt.grid(True, alpha=0.3)

        # Add statistics
        total_positions = df["count"].sum()
        weighted_mean = np.average(df["men"], weights=df["count"])

        plt.text(
            0.7,
            0.9,
            f"Total positions: {total_positions:,}\nMean pieces: {weighted_mean:.1f}",
            transform=plt.gca().transAxes,
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8),
        )

        plt.tight_layout()
        plt.savefig("material_distribution.png", dpi=300, bbox_inches="tight")
        plt.show()

    except FileNotFoundError:
        print(f"File {csv_file} not found")
    except Exception as e:
        print(f"Error plotting material distribution: {e}")


def plot_piece_square_distribution(csv_file: str = "piece_counts.csv") -> None:
    """Plot distribution of piece positions on squares."""
    try:
        df = pd.read_csv(csv_file)

        plt.figure(figsize=(14, 8))

        # Create bar plot
        bars = plt.bar(range(len(df)), df["count"], color="purple", alpha=0.7)

        plt.xlabel("Square Index")
        plt.ylabel("Count")
        plt.title("Distribution of Piece Positions Across Chess Squares")
        plt.grid(True, alpha=0.3)

        # Highlight some interesting squares if data allows
        if len(df) == 64:  # Standard chess board
            # Color center squares differently
            center_squares = [27, 28, 35, 36]  # d4, e4, d5, e5
            for i in center_squares:
                if i < len(bars):
                    bars[i].set_color("orange")
                    bars[i].set_alpha(0.9)

        plt.tight_layout()
        plt.savefig("piece_square_distribution.png", dpi=300, bbox_inches="tight")
        plt.show()

    except FileNotFoundError:
        print(f"File {csv_file} not found")
    except Exception as e:
        print(f"Error plotting piece square distribution: {e}")


def plot_king_position_distribution(csv_file: str = "pov_king_positions.csv") -> None:
    """Plot distribution of king positions from point of view."""
    try:
        df = pd.read_csv(csv_file)

        plt.figure(figsize=(14, 8))

        # Create bar plot
        plt.bar(
            range(len(df)), df["count"], color="gold", alpha=0.8, edgecolor="darkorange"
        )

        plt.xlabel("King Position Index")
        plt.ylabel("Count")
        plt.title("Distribution of King Positions (Point of View)")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("king_position_distribution.png", dpi=300, bbox_inches="tight")
        plt.show()

    except FileNotFoundError:
        print(f"File {csv_file} not found")
    except Exception as e:
        print(f"Error plotting king position distribution: {e}")


def plot_opening_evaluation_distribution(
    csv_file: str = "opening_eval_counts.csv",
) -> None:
    """Plot distribution of opening evaluations."""
    try:
        df = pd.read_csv(csv_file)

        # Convert eval to numeric if needed
        if df["eval"].dtype == "object":
            df["eval"] = pd.to_numeric(df["eval"], errors="coerce")

        # Remove any NaN values
        df = df.dropna()

        plt.figure(figsize=(14, 8))

        plt.bar(
            df["eval"],
            df["count"],
            width=1,
            alpha=0.7,
            color="teal",
            edgecolor="darkcyan",
            linewidth=0.1,
        )

        plt.xlabel("Opening Evaluation (centipawns)")
        plt.ylabel("Count")
        plt.title("Distribution of Chess Opening Evaluations")
        plt.grid(True, alpha=0.3)

        # Add vertical line at 0 (equal opening)
        plt.axvline(
            x=0, color="black", linestyle="--", alpha=0.7, label="Equal opening"
        )
        plt.legend()

        plt.tight_layout()
        plt.savefig("opening_evaluation_distribution.png", dpi=300, bbox_inches="tight")
        plt.show()

    except FileNotFoundError:
        print(f"File {csv_file} not found")
    except Exception as e:
        print(f"Error plotting opening evaluation distribution: {e}")


def create_summary_plot() -> None:
    """Create a summary plot with multiple subplots."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("Chess Data Analysis Summary", fontsize=16, fontweight="bold")

    # Plot 1: Game lengths
    try:
        df = pd.read_csv("length_counts.csv")
        axes[0, 0].plot(df["length"], df["count"], color="blue", alpha=0.7)
        axes[0, 0].fill_between(df["length"], df["count"], alpha=0.3, color="blue")
        axes[0, 0].set_title("Game Length Distribution")
        axes[0, 0].set_xlabel("Length (moves)")
        axes[0, 0].set_ylabel("Count")
        axes[0, 0].grid(True, alpha=0.3)
    except:
        axes[0, 0].text(
            0.5,
            0.5,
            "Length data\nnot available",
            ha="center",
            va="center",
            transform=axes[0, 0].transAxes,
        )

    # Plot 2: Evaluations
    try:
        df = pd.read_csv("eval_counts.csv")
        if df["eval"].dtype == "object":
            df["eval"] = pd.to_numeric(df["eval"], errors="coerce")
        df = df.dropna()
        axes[0, 1].bar(df["eval"], df["count"], width=1, alpha=0.7, color="red")
        axes[0, 1].set_title("Evaluation Distribution")
        axes[0, 1].set_xlabel("Evaluation (cp)")
        axes[0, 1].set_ylabel("Count")
        axes[0, 1].set_xlim(-1000, 1000)
        axes[0, 1].grid(True, alpha=0.3)
    except:
        axes[0, 1].text(
            0.5,
            0.5,
            "Evaluation data\nnot available",
            ha="center",
            va="center",
            transform=axes[0, 1].transAxes,
        )

    # Plot 3: Material
    try:
        df = pd.read_csv("material_counts.csv")
        axes[0, 2].plot(df["men"], df["count"], marker="o", markersize=3, color="green")
        axes[0, 2].fill_between(df["men"], df["count"], alpha=0.3, color="green")
        axes[0, 2].set_title("Material Distribution")
        axes[0, 2].set_xlabel("Pieces on Board")
        axes[0, 2].set_ylabel("Count")
        axes[0, 2].grid(True, alpha=0.3)
    except:
        axes[0, 2].text(
            0.5,
            0.5,
            "Material data\nnot available",
            ha="center",
            va="center",
            transform=axes[0, 2].transAxes,
        )

    # Plot 4: Piece squares
    try:
        df = pd.read_csv("piece_counts.csv")
        axes[1, 0].bar(range(len(df)), df["count"], color="purple", alpha=0.7)
        axes[1, 0].set_title("Piece Square Distribution")
        axes[1, 0].set_xlabel("Square Index")
        axes[1, 0].set_ylabel("Count")
        axes[1, 0].grid(True, alpha=0.3)
    except:
        axes[1, 0].text(
            0.5,
            0.5,
            "Piece square data\nnot available",
            ha="center",
            va="center",
            transform=axes[1, 0].transAxes,
        )

    # Plot 5: King positions
    try:
        df = pd.read_csv("pov_king_positions.csv")
        axes[1, 1].bar(range(len(df)), df["count"], color="gold", alpha=0.8)
        axes[1, 1].set_title("King Position Distribution")
        axes[1, 1].set_xlabel("Position Index")
        axes[1, 1].set_ylabel("Count")
        axes[1, 1].grid(True, alpha=0.3)
    except:
        axes[1, 1].text(
            0.5,
            0.5,
            "King position data\nnot available",
            ha="center",
            va="center",
            transform=axes[1, 1].transAxes,
        )

    # Plot 6: Opening evaluations
    try:
        df = pd.read_csv("opening_eval_counts.csv")
        if df["eval"].dtype == "object":
            df["eval"] = pd.to_numeric(df["eval"], errors="coerce")
        df = df.dropna()
        axes[1, 2].bar(df["eval"], df["count"], width=1, alpha=0.7, color="teal")
        axes[1, 2].set_title("Opening Evaluation Distribution")
        axes[1, 2].set_xlabel("Evaluation (cp)")
        axes[1, 2].set_ylabel("Count")
        axes[1, 2].grid(True, alpha=0.3)
    except:
        axes[1, 2].text(
            0.5,
            0.5,
            "Opening eval data\nnot available",
            ha="center",
            va="center",
            transform=axes[1, 2].transAxes,
        )

    plt.tight_layout()
    plt.savefig("chess_data_summary.png", dpi=300, bbox_inches="tight")
    plt.show()


def main():
    """Main function to generate all plots."""
    print("Setting up plot style...")
    setup_plot_style()

    csv_path = "eval_counts.csv"

    # print("Generating individual plots...")
    # plot_length_distribution()
    plot_evaluation_distribution(csv_path)
    # plot_material_distribution()
    # plot_piece_square_distribution()
    # plot_king_position_distribution()
    # plot_opening_evaluation_distribution()

    # print("Generating summary plot...")
    # create_summary_plot()

    print("All plots have been generated and saved!")


if __name__ == "__main__":
    main()
