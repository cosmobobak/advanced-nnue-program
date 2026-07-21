import os
from math import sqrt, ceil
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "TX-02"

plt.style.use("dark_background")

CHECKPOINT_DIR = "/home/cosmo/bullet/hp-sweep"
LOG_FILE = "log.txt"

# this is dependent on in-trainer config, and may need changed.
# it’s derived from taking a contingent default batches-per of
# 6104 and dividing by four.
BATCHES_PER_SB = 1526
# similarly so.
BATCH_INCREMENT = 32

SUFFIXES: dict[str, str] = {
    "-s0": "Stage 0",
    "-s1": "Stage 1",
    "-s2": "Stage 2",
}

# TESTS: list[str] = [
#     "excitement-0.2-lr1",
#     "excitement-0.2-lr0.5",
#     "excitement-0.2-lr0.25",
#     "excitement-0.2-lr0.125",
#     "excitement-0.2-lr2",
#     "excitement-0.2-lr4",
# ]

TESTS: list[str] = [
    "juice-0.05",
    "juice-0.1",
    "juice-0.2",
    "juice-0.5",
    "juice-1",
    "juice-2",
    # "juice-4",
]

BASELINE: str | None = None
# BASELINE: str | None = TESTS[0]

EMA_ALPHA: float = 0.005

SKIP_DATAPOINTS: int = 50

SHOW_RAW: bool = True

RENORMALISE_FRACTIONAL: bool = False

# the log format is <superbatch>:<batch-within-superbatch>:<loss>

# we want to, for each SUFFIX, construct a subplot.

# to do this, we make a map from suffix to fileset
# e.g.
# ```json
# {
#   "Stage 0": { "meow": "~/bullet/checkpoints/meow-s0", "love": "~/bullet/checkpoints/love-s0" },
#   "Stage 1": { "meow": "~/bullet/checkpoints/meow-s1", "love": "~/bullet/checkpoints/love-s1" },
#   &c &c
# }
# ```
FILE_MAP: dict[str, dict[str, str]] = {
    SUFFIXES[suffix]: {f"{test}": f"{test}{suffix}" for test in TESTS}
    for suffix in SUFFIXES
}


def find_match(file_substring: str) -> str:
    return next(
        f"{CHECKPOINT_DIR}/{file}"
        for file in sorted(os.listdir(CHECKPOINT_DIR), reverse=True)
        if file_substring in file
    )


# read a logfile and convert it into absolute batch : loss traces.
def parse(
    file_substring: str,
) -> tuple[np.ndarray, np.ndarray]:
    indexes = []
    losses = []
    file = f"{find_match(file_substring)}/{LOG_FILE}"
    print(f"READING {file}")
    for line in open(file, "r", encoding="utf-8").readlines():
        sb, b, loss = line.split(",")
        sb, b = int(sb) - 1, int(b) - 32
        loss = float(loss)
        batch = BATCHES_PER_SB * sb + b
        indexes.append(batch)
        losses.append(loss)
    print(f"N: {len(indexes)} SB: {len(indexes) / 47}")
    indexes = indexes[SKIP_DATAPOINTS:]
    losses = losses[SKIP_DATAPOINTS:]
    return np.array(indexes), np.array(losses)


DATA_MAP: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {
    stage: {name: parse(path) for name, path in files.items()}
    for stage, files in FILE_MAP.items()
}

# square plots are nicer, so we choose a side length
# close to the sqrt of the stage count.
PLOT_SIDE_LEN: int = ceil(sqrt(len(SUFFIXES)))


def ema(x: np.ndarray, alpha: float = EMA_ALPHA) -> np.ndarray:
    out = np.empty_like(x)
    acc = x[0]
    for i, v in enumerate(x):
        acc = alpha * v + (1 - alpha) * acc
        out[i] = acc
    return out


def draw_single_curve(subplot, name: str, indexes: np.ndarray, losses: np.ndarray):
    indexes = indexes / BATCHES_PER_SB
    if RENORMALISE_FRACTIONAL:
        indexes -= min(indexes)
        indexes /= max(indexes)
    if SHOW_RAW:
        (raw,) = subplot.plot(
            indexes,
            losses,
            alpha=0.2,
            lw=0.8,
            label="_nolegend_",
        )
        subplot.plot(
            indexes,
            ema(losses),
            color=raw.get_color(),
            lw=1.5,
            label=name,
            zorder=3,
        )
    else:
        subplot.plot(indexes, ema(losses), label=name)


def draw_absolute(subplot, stage: str, data: dict[str, tuple[np.ndarray, np.ndarray]]):
    assert BASELINE is None

    # some clipping to make extreme values not ruin the plot:
    ys = np.concatenate([losses for _, losses in data.values()])
    lo_y, hi_y = np.percentile(ys, [0.5, 99.5])
    subplot.set_ylim(lo_y - 0.02 * (hi_y - lo_y), hi_y + 0.02 * (hi_y - lo_y))

    for name, (indexes, losses) in data.items():
        draw_single_curve(subplot, name, indexes, losses)

    subplot.set_ylabel("loss")


def draw_baseline(subplot, stage: str, data: dict[str, tuple[np.ndarray, np.ndarray]]):
    assert BASELINE is not None

    base_idx, base_loss = data[BASELINE]

    # clip to overlapping region
    lo = max(idx[0] for idx, _ in data.values())
    hi = min(idx[-1] for idx, _ in data.values())
    mask = (base_idx >= lo) & (base_idx <= hi)
    grid = base_idx[mask]
    base_on_grid = base_loss[mask]

    for name, (indexes, losses) in data.items():
        if name == BASELINE:
            continue
        delta = np.interp(grid, indexes, losses) - base_on_grid
        draw_single_curve(subplot, name, grid, delta)

    subplot.axhline(0.0, color="w", lw=0.8, ls="--", label=f"{BASELINE} (baseline)")
    subplot.set_ylabel(f"loss − {BASELINE}")


def draw(subplot, stage: str, data: dict[str, tuple[np.ndarray, np.ndarray]]):
    subplot.set_prop_cycle(color=plt.cm.tab10.colors)  # type: ignore

    if BASELINE is None:
        draw_absolute(subplot, stage, data)
    else:
        draw_baseline(subplot, stage, data)

    subplot.grid(False)
    subplot.set_title(stage)
    subplot.set_xlabel("superbatch")
    subplot.legend(fontsize="small")
    subplot.ticklabel_format(axis="x", style="sci", scilimits=(0, 4))


def main():
    nrows = ceil(len(SUFFIXES) / PLOT_SIDE_LEN)
    fig, axes = plt.subplots(
        nrows, PLOT_SIDE_LEN, layout="constrained", figsize=(12, 7)
    )
    axs = list(axes.flat)

    for ax, (stage, data) in zip(axs, DATA_MAP.items()):
        draw(ax, stage, data)
    for ax in axs[len(DATA_MAP) :]:  # hide unused cells
        ax.set_visible(False)

    plt.show()
    plt.savefig("opt-plot.svg")
    plt.savefig("opt-plot.png")


if __name__ == "__main__":
    main()
