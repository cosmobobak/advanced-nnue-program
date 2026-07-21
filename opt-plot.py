import os
from math import sqrt, ceil, floor, log10
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator, NullLocator, ScalarFormatter

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
# datapoints logged per superbatch.
# would be BATCHES_PER_SB / BATCH_INCREMENT
# but grouping is imperfect.
DATAPOINTS_PER_SB = 47

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

EMA_ALPHA: float = 0.05

SKIP_DATAPOINTS: int = 50

SHOW_RAW: bool = True

RENORMALISE_FRACTIONAL: bool = False

LOG_Y: bool = True

CONTINUOUS_PALETTE: bool = True
CONTINUOUS_CMAP: str = "inferno"

# the log format is <superbatch>,<batch-within-superbatch>,<loss>

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
        sb, b = int(sb) - 1, int(b) - BATCH_INCREMENT
        loss = float(loss)
        batch = BATCHES_PER_SB * sb + b
        indexes.append(batch)
        losses.append(loss)
    print(f"N: {len(indexes)} SB: {len(indexes) / DATAPOINTS_PER_SB}")
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


def nice_step(span: float, target: int = 12, steps=(1, 2, 2.5, 5, 10)) -> float:
    raw = span / target
    mag = 10.0 ** floor(log10(raw))
    return next(s * mag for s in steps if s * mag >= raw)


def plain_decimals(axis):
    for set_fmt in (axis.set_major_formatter, axis.set_minor_formatter):
        fmt = ScalarFormatter()
        fmt.set_scientific(False)
        fmt.set_useOffset(False)
        set_fmt(fmt)


def palette(n: int) -> list:
    if CONTINUOUS_PALETTE:
        cmap = plt.get_cmap(CONTINUOUS_CMAP)
        return [cmap(t) for t in np.linspace(0.15, 1.0, n)]
    return list(plt.cm.tab10.colors[:n])  # type: ignore


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
    lo_y, hi_y = np.percentile(ys, [0.001, 99.99])
    margin = 0.02 * (hi_y - lo_y)
    lo_y, hi_y = lo_y - margin, hi_y + margin

    if LOG_Y:
        step = nice_step(hi_y - lo_y)
        lo_y, hi_y = floor(lo_y / step) * step, ceil(hi_y / step) * step
        subplot.yaxis.set_major_locator(MultipleLocator(step))

    subplot.set_ylim(lo_y, hi_y)

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
    subplot.set_prop_cycle(color=palette(len(data)))

    if BASELINE is None:
        if LOG_Y:
            subplot.set_yscale("log")
            subplot.yaxis.set_minor_locator(NullLocator())
            plain_decimals(subplot.yaxis)
        draw_absolute(subplot, stage, data)
    else:
        draw_baseline(subplot, stage, data)

    subplot.grid(False)
    subplot.set_title(stage)
    subplot.set_xlabel("superbatch")
    subplot.ticklabel_format(axis="x", style="sci", scilimits=(0, 4))


def main():
    nrows = ceil(len(SUFFIXES) / PLOT_SIDE_LEN)
    fig, axes = plt.subplots(
        nrows, PLOT_SIDE_LEN, layout="constrained", figsize=(12, 7)
    )
    axs = list(axes.flat)

    for ax, (stage, data) in zip(axs, DATA_MAP.items()):
        draw(ax, stage, data)

    handles, labels = axs[0].get_legend_handles_labels()
    spare = axs[len(DATA_MAP) :]
    for ax in spare[1:]:  # hide any fully-unused cells
        ax.set_visible(False)
    if spare:
        # shared legend in the first empty cell
        spare[0].axis("off")
        spare[0].legend(handles, labels, loc="center", fontsize="large")
    else:
        fig.legend(handles, labels, loc="outside right upper", fontsize="small")

    # plt.show()
    # plt.savefig("opt-plot.svg", bbox_inches="tight", pad_inches=0.4)
    plt.savefig("opt-plot.png", bbox_inches="tight", pad_inches=0.4)


if __name__ == "__main__":
    main()
