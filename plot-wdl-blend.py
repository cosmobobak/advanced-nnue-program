import matplotlib.pyplot as plt
import numpy as np

from qbstyles import mpl_style
mpl_style(dark=True)

# Extract data from the test results
versions = ['basilisk.4', 'basilisk.5', 'basilisk.6', 'basilisk.7', 'basilisk.8', 'basilisk.9']
version_numbers = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Fixed-nodes Elo values and errors (95% confidence intervals)
fixed_nodes_elo = [0.71, -0.31, -4.85, -13.95, -19.27, -26.61]
fixed_nodes_err = [0.75, 1.76, 3.74, 5.93, 7.09, 8.26]

# LTC Elo values and errors (missing data for basilisk.8 and basilisk.9)
ltc_version_numbers = [0.4, 0.5, 0.6, 0.7, 0.8]
ltc_elo = [-8.80, -7.40, -5.47, -5.77, -10.67]
ltc_err = [4.35, 4.00, 3.55, 3.59, 4.86]

# Create the plot
plt.figure(figsize=(16, 12))

# Plot fixed-nodes data with error bars
plt.errorbar(version_numbers, fixed_nodes_elo, yerr=fixed_nodes_err, 
             marker='o', capsize=5, capthick=2, label='Fixed-nodes', 
             color='coral', linewidth=2, markersize=8, elinewidth=2)

# Plot LTC data with error bars
plt.errorbar(ltc_version_numbers, ltc_elo, yerr=ltc_err, 
             marker='s', capsize=5, capthick=2, label='LTC (40.0+0.40s)', 
             color='cyan', linewidth=2, markersize=8, elinewidth=2)

# Customize the plot
plt.xlabel('WDL blend', fontsize=12)
plt.ylabel('Elo', fontsize=12)
plt.title('Fixed-nodes vs LTC\n(vs. delenda, 95% confidence intervals)', 
          fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(fontsize=14, loc='upper right')

# Add a horizontal line at y=0 for reference
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)

# Set x-axis to show integer version numbers
plt.xticks(version_numbers)

# Add some padding to the y-axis
y_min = min(min(fixed_nodes_elo) - max(fixed_nodes_err), min(ltc_elo) - max(ltc_err)) - 5
y_max = max(max(fixed_nodes_elo) + max(fixed_nodes_err), max(ltc_elo) + max(ltc_err)) + 5
plt.ylim(y_min, y_max)

# Add text annotations for missing LTC data
# plt.text(8, -15, 'LTC data\nnot available', fontsize=10, 
#          ha='center', va='center', style='italic', color='gray',
#          bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='gray', alpha=0.7))

plt.tight_layout()
plt.show()

# Print summary statistics
print("Fixed-nodes Elo progression (vs. delenda):")
print("-" * 50)
for i, v in enumerate(versions):
    print(f"{v}: {fixed_nodes_elo[i]:>6.2f} ± {fixed_nodes_err[i]:.2f}")
    print(f"          Games: see test links for details")

print("\nLTC Elo progression (vs. delenda):")
print("-" * 50)
for i, v in enumerate(versions[:4]):
    print(f"{v}: {ltc_elo[i]:>6.2f} ± {ltc_err[i]:.2f}")
print("basilisk.8: No data")
print("basilisk.9: No data")

# Calculate and print trends
print("\nTrend Analysis:")
print("-" * 50)
print(f"Fixed-nodes: {fixed_nodes_elo[0]:.2f} → {fixed_nodes_elo[-1]:.2f} "
      f"(Δ = {fixed_nodes_elo[-1] - fixed_nodes_elo[0]:.2f} Elo)")
print(f"LTC (v4-v7): {ltc_elo[0]:.2f} → {ltc_elo[-1]:.2f} "
      f"(Δ = {ltc_elo[-1] - ltc_elo[0]:.2f} Elo)")

plt.savefig("wdl-blend.png")
