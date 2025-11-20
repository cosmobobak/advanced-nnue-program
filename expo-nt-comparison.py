# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme()

# %%
viri_data = {
    "total_nodes": 337825314.0,
    "full_window": 181392.0,
    "expected_pv": 181392.0,
    "first_cut": 57290.0,
    "first_meet": 66546.0,
    "first_fail": 57556.0,
    "first_fail_later_cut": 9530.0,
    "first_fail_all_fail": 35424.0,
    "first_fail_later_pv": 12501.0,
}

expo_data = {
    "total_nodes": 228300000.0,
    "full_window": 26534.0,
    "expected_pv": 8496.0,
    "first_cut": 1833.0,
    "first_meet": 5577.0,
    "first_fail": 1086.0,
    "first_fail_later_cut": 117.0,
    "first_fail_all_fail": 26.0,
    "first_fail_later_pv": 943.0,
}

# 98213 fw                   28755 fw pv
# 26867 fw first cut          6350 fw pv first cut
# 51242 fw first meet        18527 fw pv first meet
# 20104 fw first fail         3878 fw pv first fail
#  2310 fw first fail cut      465 fw pv first fail cut
#  6743 fw first fail all     1086 fw pv first fail all

expov2_data = {
    "total_nodes": 0.0,
    "full_window": 98213.0,
    "expected_pv": 8496.0,
    "first_cut": 1833.0,
    "first_meet": 5577.0,
    "first_fail": 1086.0,
    "first_fail_later_cut": 117.0,
    "first_fail_all_fail": 26.0,
    "first_fail_later_pv": 943.0,
}

vff = viri_data["first_fail"]
eff = expo_data["first_fail"]

# %%
# normalise first_{cut,meet,fail} by expected_pv
for key in ["first_cut", "first_meet", "first_fail"]:
    viri_data[key] /= viri_data["expected_pv"]
    expo_data[key] /= expo_data["expected_pv"]

# normalise first_fail_{later_cut,all_fail,later_pv} by first_fail
for key in ["first_fail_later_cut", "first_fail_all_fail", "first_fail_later_pv"]:
    viri_data[key] /= vff
    expo_data[key] /= eff

# %%
# Plot 1: Compare first_{cut,meet,fail} ratios
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

categories_1 = ["first_cut", "first_meet", "first_fail"]
viri_values_1 = [viri_data[k] for k in categories_1]
expo_values_1 = [expo_data[k] for k in categories_1]

x = np.arange(len(categories_1))
width = 0.35

ax1.bar(x - width / 2, viri_values_1, width, label="Viri", alpha=0.8)
ax1.bar(x + width / 2, expo_values_1, width, label="Expo", alpha=0.8)
ax1.set_xlabel("node type")
ax1.set_ylabel("fraction of total")
ax1.set_title("pv first move")
ax1.set_xticks(x)
ax1.set_xticklabels(["cut", "meet", "fail"])
ax1.legend()
ax1.grid(axis="y", alpha=0.3)

# Plot 2: Compare first_fail_{later_cut,all_fail,later_pv} ratios
categories_2 = ["first_fail_later_cut", "first_fail_all_fail", "first_fail_later_pv"]
viri_values_2 = [viri_data[k] for k in categories_2]
expo_values_2 = [expo_data[k] for k in categories_2]

x2 = np.arange(len(categories_2))

ax2.bar(x2 - width / 2, viri_values_2, width, label="Viri", alpha=0.8)
ax2.bar(x2 + width / 2, expo_values_2, width, label="Expo", alpha=0.8)
ax2.set_xlabel("outcome after first fail")
ax2.set_ylabel("fraction of total")
ax2.set_title("first fail outcomes")
ax2.set_xticks(x2)
ax2.set_xticklabels(["cut", "all fail", "PV"])
ax2.legend()
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("expo_nt_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nnode type:")
for key in categories_1:
    print(f"{key:15s}: Viri={viri_data[key]:.4f}, Expo={expo_data[key]:.4f}")

print("\nfirst fail:")
for key in categories_2:
    print(f"{key:25s}: Viri={viri_data[key]:.4f}, Expo={expo_data[key]:.4f}")
