"""D* Lite Replanning

Repair a route after a doorway closes while the robot is already partway along its journey.

Koenig and Likhachev, D* Lite (2002)
https://publications.ri.cmu.edu/d-lite"""

from pathlib import Path
import sys

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parent))
import numpy as np
import matplotlib.pyplot as plt

from planning import DStarLite, grid_search


def main():
    grid = np.zeros((18, 30), bool)
    grid[:, 14] = True
    grid[4, 14] = False
    grid[13, 14] = False
    start = (4, 2)
    goal = (4, 27)
    planner = DStarLite(grid, start, goal)
    first_exp = planner.repair()
    before = planner.path()
    assert (4, 14) in before
    moved = before[5]
    planner.move_start(moved)
    planner.change_cell((4, 14), True)
    updates = planner.repair()
    after = planner.path()
    ref, _ = grid_search(planner.grid, moved, goal)
    assert (
        len(after) == len(ref) and (13, 14) in after and all(not planner.grid[p] for p in after)
    )
    planner.change_cell((4, 14), False)
    planner.repair()
    reopened = planner.path()
    assert len(reopened) == len(grid_search(grid, moved, goal)[0])
    print(
        "Original steps",
        len(before) - 1,
        "after closure",
        len(after) - 1,
        "after reopening",
        len(reopened) - 1,
    )
    print("Queue pops on initial plan and repair", first_exp, updates)
    fig, ax = plt.subplots(figsize=(6.9, 3.3), constrained_layout=True)
    ax.imshow(grid, cmap="Greys", vmin=0, vmax=1)
    for points, label, style in [(before, "Original", "--"), (after, "Repaired", "-")]:
        y, x = np.array(points).T
        ax.plot(x, y, style, label=label)
    ax.scatter(14, 4, marker="x", s=70, color="firebrick", label="Closed doorway")
    ax.set(xlabel="Column", ylabel="Row")
    ax.legend(fontsize=8)
    plt.show()


if __name__ == "__main__":
    main()
