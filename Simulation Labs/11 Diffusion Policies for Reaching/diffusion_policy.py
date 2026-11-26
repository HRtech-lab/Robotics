"""Diffusion Policies for Reaching

Learn short actions for a simulated tabletop point moving towards a goal from offline demonstrations.

Chi et al., Diffusion Policy (2023)
https://diffusion-policy.cs.columbia.edu/
Song, Meng and Ermon, Denoising Diffusion Implicit Models
https://arxiv.org/abs/2010.02502"""

from pathlib import Path
import sys

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parent))
import numpy as np
import matplotlib.pyplot as plt

import torch
from reaching import fit_policy, evaluate


def main():
    kind = "diffusion"
    model, alpha, losses = fit_policy(kind, seed=31, steps=1250)
    print(
        "Gradient updates",
        len(losses),
        "| mean objective, first and last 100",
        np.round([losses[:100].mean(), losses[-100:].mean()], 5),
    )

    paths, goals, chunk_error = evaluate(model, alpha, kind, seed=197)
    distance = np.linalg.norm(paths - goals[None, :, :], axis=2)
    success = np.mean(distance[-1] < 0.08)
    print(
        f"Action RMSE {chunk_error:.4f} m/tick | mean distance before {distance[0].mean():.3f} m | after {distance[-1].mean():.3f} m"
    )
    print(f"Within 8 cm: {int((distance[-1]<.08).sum())} of {len(goals)}")
    assert distance[-1].mean() < 0.35 * distance[0].mean()
    fig, (a, b) = plt.subplots(1, 2, figsize=(8, 3.45), constrained_layout=True)
    for i in range(8):
        line = a.plot(*paths[:, i, :].T, alpha=0.82)[0]
        a.scatter(*goals[i], marker="x", color=line.get_color())
    a.set(aspect="equal", xlabel="x (m)", ylabel="y (m)", title="Eight held-out trajectories")
    steps = np.arange(len(distance))
    b.fill_between(
        steps,
        *np.quantile(distance, [0.1, 0.9], axis=1),
        alpha=0.22,
        label="10th–90th percentile",
    )
    b.plot(steps, distance.mean(1), label="Mean")
    b.set(xlabel="Executed actions", ylabel="Distance to goal (m)")
    b.legend(fontsize=7)
    plt.show()


if __name__ == "__main__":
    main()
