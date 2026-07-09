# /// script
# requires-python = ">=3.14"
# dependencies = ["matplotlib"]
# ///

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "TeX Gyre Heros"

data = """
men,count
3,154670260
4,488606995
5,714646136
6,1170960677
7,1370836785
8,1297789902
9,1366172922
10,1215472624
11,1219254671
12,1107026743
13,1069858870
14,1005002439
15,952838462
16,924978459
17,861924944
18,867803204
19,805020251
20,837865626
21,765371961
22,836524457
23,745683861
24,856888184
25,731122401
26,911662443
27,735126307
28,1006449650
29,756533835
30,1198476995
31,774584861
32,1326420926
"""

men, count = zip(*[map(int, line.split(",")) for line in data.strip().splitlines()[1:]])

men: list[int] = men  # type: ignore
count: list[int] = count  # type: ignore

total: int = sum(count)
count: list[float] = [c / total for c in count]

# set max to 1.0.
max_count: float = max(count)
count = [c / max_count for c in count]


def weight_for_count(count: int) -> float:
    x = count
    x1, y1 = 0, 0.75
    x2, y2 = 16, 1.0
    x3, y3 = 32, 0.75
    l1: float = (x - x2) * (x - x3) / ((x1 - x2) * (x1 - x3))
    l2: float = (x - x1) * (x - x3) / ((x2 - x1) * (x2 - x3))
    l3: float = (x - x1) * (x - x2) / ((x3 - x1) * (x3 - x2))
    return l1 * y1 + l2 * y2 + l3 * y3


weights = [weight_for_count(m) for m in men]

# compute p(retain) for each piece count s.t. count_i * p(retain)_i = weights_i
retain_probs = [w / c if c > 0 else 0.0 for w, c in zip(weights, count)]
# probabilities cannot exceed 1.0, as we can't create more data than we have.
max_retain_prob = max(retain_probs)
retain_probs = [rp / max_retain_prob for rp in retain_probs]

fig, ax = plt.subplots(figsize=(8, 8))

# ax.plot(men, count, color="gray", linewidth=1.5)
# ax.scatter(men, count, color="gray", s=20, zorder=5)
ax.bar(men, count, color="black")
ax.plot(men, weights, color="magenta", linewidth=1.5, label="desired weight")
ax.scatter(men, weights, color="magenta", s=20, zorder=5)
ax.plot(men, retain_probs, color="grey", linewidth=1.5, label="implied p(retain)")
ax.scatter(men, retain_probs, color="grey", s=20, zorder=5)
ax.legend(fontsize=12)

ax.set_xlabel("pieces", fontsize=12)
ax.set_ylabel("proportion", fontsize=12)

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

# ax.set_xlim(-0.05, 1.05)
# ax.set_ylim(-30, 6)
plt.axhline(y=0, color="gray", linestyle="--", alpha=0.5, linewidth=1)

plt.tight_layout()
plt.savefig("scale_plot.png", dpi=150, facecolor="white", edgecolor="none")
plt.show()
