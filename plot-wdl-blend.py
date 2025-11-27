import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style("whitegrid")

# from qbstyles import mpl_style

# mpl_style(dark=True)

# Extract data from the test results
versions = [
    "λ=0.0",
    "λ=0.1",
    "λ=0.2",
    "λ=0.3",
    "λ=0.4",
    "λ=0.5",
    "λ=0.6",
    "λ=0.7",
    "λ=0.8",
    "λ=0.9",
    "λ=1.0",
]
version_numbers = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Fixed-nodes Elo values and errors (95% confidence intervals)
fixed_nodes_elo = [
    -19.82,
    -16.00,
    -17.20,
    -6.87,
    0.0,
    # -7.48,
    -3.86,
    -1.85,
    -3.98,
    -13.84,
    -15.10,
    -27.80,
]
fixed_nodes_err = [
    2.37,
    2.34,
    2.33,
    2.33,
    0.0,
    # 2.31,
    2.36,
    2.36,
    2.36,
    2.40,
    2.41,
    2.42,
]

# LTC Elo values and errors (95% confidence intervals)
# ltc_version_numbers = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# ltc_elo = [-8.80, -7.40, -5.47, -5.77, -10.67, -22.81]
# ltc_err = [4.35, 4.00, 3.55, 3.59, 4.86, 6.95]

# Create the plot
plt.figure(figsize=(16, 12))

# Plot fixed-nodes data with error bars
plt.errorbar(
    version_numbers,
    fixed_nodes_elo,
    yerr=fixed_nodes_err,
    marker="o",
    capsize=5,
    capthick=2,
    label="Fixed-nodes",
    color="coral",
    linewidth=1,
    markersize=8,
    elinewidth=2,
)

# Plot LTC data with error bars
# plt.errorbar(ltc_version_numbers, ltc_elo, yerr=ltc_err,
#              marker='s', capsize=5, capthick=2, label='LTC (40.0+0.40s)',
#              color='cyan', linewidth=2, markersize=8, elinewidth=2)

# Add quadratic fit for fixed-nodes data
fixed_nodes_fit = np.polyfit(version_numbers, fixed_nodes_elo, 2)
fixed_nodes_poly = np.poly1d(fixed_nodes_fit)
x_smooth_fixed = np.linspace(
    min(version_numbers) - 0.1, max(version_numbers) + 0.1, 100
)
y_smooth_fixed = fixed_nodes_poly(x_smooth_fixed)
plt.plot(
    x_smooth_fixed,
    y_smooth_fixed,
    "--",
    color="coral",
    linewidth=2.5,
    alpha=0.4,
    label="Fixed-nodes quadratic fit",
)

# Add quadratic fit for LTC data
# ltc_fit = np.polyfit(ltc_version_numbers, ltc_elo, 2)
# ltc_poly = np.poly1d(ltc_fit)
# x_smooth_ltc = np.linspace(min(ltc_version_numbers) - 0.1, max(ltc_version_numbers) + 0.1, 100)
# y_smooth_ltc = ltc_poly(x_smooth_ltc)
# plt.plot(x_smooth_ltc, y_smooth_ltc, '--', color='dodgerblue',
#          linewidth=2.5, alpha=0.2, label='LTC quadratic fit')

# Customize the plot
plt.xlabel("WDL blend", fontsize=12)
plt.ylabel("Elo", fontsize=12)
plt.title(
    "WDL lambda hyperparameter sweep",
    fontsize=14,
    fontweight="bold",
)
plt.grid(True, alpha=0.3, linestyle="--")
plt.legend(fontsize=14, loc="upper right")

# Add a horizontal line at y=0 for reference
plt.axhline(y=0, color="gray", linestyle="--", alpha=0.5, linewidth=1)

# Set x-axis to show integer version numbers
plt.xticks(version_numbers)

# Add some padding to the y-axis
y_min = min(fixed_nodes_elo) - max(fixed_nodes_err) - 5
y_max = max(fixed_nodes_elo) + max(fixed_nodes_err) + 5
# y_min = min(min(fixed_nodes_elo) - max(fixed_nodes_err), min(ltc_elo) - max(ltc_err)) - 5
# y_max = max(max(fixed_nodes_elo) + max(fixed_nodes_err), max(ltc_elo) + max(ltc_err)) + 5
plt.ylim(y_min, y_max)

# Add text annotations for missing LTC data
# plt.text(8, -15, 'LTC data\nnot available', fontsize=10,
#          ha='center', va='center', style='italic', color='gray',
#          bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='gray', alpha=0.7))

plt.tight_layout()
plt.show()

# Print summary statistics
print("Fixed-nodes Elo:")
print("-" * 50)
for i, v in enumerate(versions):
    print(f"{v}: {fixed_nodes_elo[i]:>6.2f} ± {fixed_nodes_err[i]:.2f}")

# print("\nLTC Elo progression (vs. delenda):")
# print("-" * 50)
# for i, v in enumerate(versions[:5]):
#     print(f"{v}: {ltc_elo[i]:>6.2f} ± {ltc_err[i]:.2f}")
# print("basilisk.9: No data")

# Calculate and print trends
print("\nTrend Analysis:")
print("-" * 50)
print(
    f"Fixed-nodes: {fixed_nodes_elo[0]:.2f} → {fixed_nodes_elo[-1]:.2f} "
    f"(Δ = {fixed_nodes_elo[-1] - fixed_nodes_elo[0]:.2f} Elo)"
)
# print(
#     f"LTC (v4-v8): {ltc_elo[0]:.2f} → {ltc_elo[-1]:.2f} "
#     f"(Δ = {ltc_elo[-1] - ltc_elo[0]:.2f} Elo)"
# )

# Print quadratic fit equations
print("\nQuadratic Fit Equations:")
print("-" * 50)
print(
    f"Fixed-nodes: y = {fixed_nodes_fit[0]:.2f}x² + {fixed_nodes_fit[1]:.2f}x + {fixed_nodes_fit[2]:.2f}"
)
# print(f"LTC: y = {ltc_fit[0]:.2f}x² + {ltc_fit[1]:.2f}x + {ltc_fit[2]:.2f}")

# Calculate R-squared values
fixed_nodes_r2 = 1 - (
    np.sum((fixed_nodes_elo - fixed_nodes_poly(version_numbers)) ** 2)
    / np.sum((fixed_nodes_elo - np.mean(fixed_nodes_elo)) ** 2)  # type: ignore
)
# ltc_r2 = 1 - (
#     np.sum((ltc_elo - ltc_poly(ltc_version_numbers)) ** 2)
#     / np.sum((ltc_elo - np.mean(ltc_elo)) ** 2)
# )

print(f"\nFixed-nodes R² = {fixed_nodes_r2:.4f}")
# print(f"LTC R² = {ltc_r2:.4f}")

plt.savefig("wdl-blend.png")
