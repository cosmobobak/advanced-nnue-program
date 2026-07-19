import os
from math import sqrt, ceil
import matplotlib.pyplot as plt
import numpy as np

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
def find_batch_per(file_substring: str) -> int:
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
    return max_batch + increment


BATCHES_PER_SB = find_batch_per(
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
    return np.array(indexes), np.array(losses)


DATA_MAP: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {
    stage: {name: parse(path) for name, path in files.items()}
    for stage, files in FILE_MAP.items()
}

# square plots are nicer, so we choose a side length
# close to the sqrt of the stage count.
PLOT_SIDE_LEN: int = ceil(sqrt(len(SUFFIXES)))


def draw(subplot, stage: str, data: dict[str, tuple[np.ndarray, np.ndarray]]):

    for name, (indexes, losses) in data.items():
        subplot.plot(indexes / BATCHES_PER_SB, losses, label=name)

    subplot.set_title(stage)
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
