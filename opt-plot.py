import os
from math import sqrt, ceil
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "TX-02"

CHECKPOINT_DIR = "/home/cosmo/bullet/checkpoints"
LOG_FILE = "log.txt"

SUFFIXES: dict[str, str] = {
    "-s0": "Stage 0",
    "-s1": "Stage 1",
    "-s2": "Stage 2",
}

TESTS: list[str] = [
    "sandhi",
    "sapient",
    "bicameral",
    "circumflex",
    "magnetar",
]

# BASELINE: str | None = None
BASELINE: str | None = TESTS[0]

EMA_ALPHA: float = 0.005

SKIP_SUPERBATCHES: int = 0

SHOW_RAW: bool = False

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


# figure out how many batches run within each superbatch so we’re robust to batch size edits
def find_batch_per(file_substring: str) -> tuple[int, int]:
    max_batch = 0
    increment = None
    file = f"{find_match(file_substring)}/{LOG_FILE}"
    for i, line in enumerate(open(file, "r", encoding="utf-8").readlines()):
        sb, b, loss = line.split(",")
        sb, b = int(sb), int(b)
        # read file ’til superbatch = 1
        if sb == 2:
            break
        max_batch = max(max_batch, b)
        if i == 1:
            increment = b
    assert increment is not None, f"no lines in {file}"
    return max_batch + increment, increment


BATCHES_PER_SB, LOG_INCREMENT = find_batch_per(
    next(next(path for path in path_list.values()) for path_list in FILE_MAP.values())
)


# read a logfile and convert it into absolute batch : loss traces.
def parse(
    file_substring: str,
) -> tuple[np.ndarray, np.ndarray]:
    indexes = []
    losses = []
    file = f"{find_match(file_substring)}/{LOG_FILE}"
    for line in open(file, "r", encoding="utf-8").readlines():
        sb, b, loss = line.split(",")
        sb, b = int(sb), int(b)
        loss = float(loss)
        batch = BATCHES_PER_SB * sb + b
        indexes.append(batch)
        losses.append(loss)
    indexes = indexes[(SKIP_SUPERBATCHES * BATCHES_PER_SB // LOG_INCREMENT) :]
    losses = losses[(SKIP_SUPERBATCHES * BATCHES_PER_SB // LOG_INCREMENT) :]
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
    if SHOW_RAW:
        (raw,) = subplot.plot(
            indexes / BATCHES_PER_SB,
            losses,
            alpha=0.2,
            lw=0.8,
            label="_nolegend_",
        )
        subplot.plot(
            indexes / BATCHES_PER_SB,
            ema(losses),
            color=raw.get_color(),
            lw=1.5,
            label=name,
            zorder=3,
        )
    else:
        subplot.plot(indexes / BATCHES_PER_SB, ema(losses), label=name)


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

    subplot.axhline(0.0, color="black", lw=0.8, ls="--", label=f"{BASELINE} (baseline)")
    subplot.set_ylabel(f"loss − {BASELINE}")


def draw(subplot, stage: str, data: dict[str, tuple[np.ndarray, np.ndarray]]):
    subplot.set_prop_cycle(color=plt.cm.tab10.colors)  # type: ignore

    if BASELINE is None:
        draw_absolute(subplot, stage, data)
    else:
        draw_baseline(subplot, stage, data)

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


if __name__ == "__main__":
    main()
