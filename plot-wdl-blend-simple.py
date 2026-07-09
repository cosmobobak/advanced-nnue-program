# /// script
# requires-python = ">=3.14"
# dependencies = ["matplotlib"]
# ///

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "TeX Gyre Heros"

lambda_vals: list[float] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# scale_vals: list[float] = [484, 469, 448, 429, 414, 389, 365, 343, 315, 287, 229]
elo_vals: list[float] = [
    -19.82,
    -16.00,
    -17.20,
    -6.87,
    0.0,
    -3.86,
    -1.85,
    -3.98,
    -13.84,
    -15.10,
    -27.80,
]
elo_vals_scaled: list[float] = [
    -21.82,
    -20.92,
    -16.65,
    -5.62,
    0.0,
    -1.41,
    0.67,
    4.42,
    0.88,
    3.06,
    0.36,
]

fig, ax = plt.subplots(figsize=(8, 8))

ax.plot(lambda_vals, elo_vals, color="gray", linewidth=1.5)
ax.scatter(lambda_vals, elo_vals, color="gray", s=20, zorder=5)

ax.plot(lambda_vals, elo_vals_scaled, color="black", linewidth=1.5)
ax.scatter(lambda_vals, elo_vals_scaled, color="black", s=20, zorder=5)

ax.set_xlabel("λ", fontsize=12)
ax.set_ylabel("Elo", fontsize=12)

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

ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-30, 6)
plt.axhline(y=0, color="gray", linestyle="--", alpha=0.5, linewidth=1)

plt.tight_layout()
plt.savefig("scale_plot.png", dpi=150, facecolor="white", edgecolor="none")
plt.show()
