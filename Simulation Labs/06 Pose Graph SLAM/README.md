# Pose Graph SLAM

Correct accumulated odometry drift using relative-pose constraints from repeated places.

Each edge describes a relative translation and rotation in the earlier pose frame. Translation and angle errors are divided by their measurement standard deviations. The first pose stays fixed to resolve the unobservable global frame. A robust loss reduces the influence of an inconsistent loop constraint.

[Worked notebook](pose_graph_slam.ipynb) · [Python script](pose_graph_slam.py)

## Optimise a loop with a known closure

The known scene provides loop associations. Compare odometry, least squares and Huber estimation against the held-out trajectory coordinates.

## Recorded output

```text
Position RMSE (m), odometry / least squares / Huber [0.482 0.275 0.166]
Closure gap (m) [0.744 0.06  0.057]
```

## Interpreting the result

The map is a graph of poses. Reliable closure detection is a separate front-end problem. Huber loss acts on the whitened scalar residuals in this implementation.

![Pose Graph SLAM, worked result](figure.svg)

## Exercise

Remove the bad closure, then compare both estimation objectives again.

## References

[Grisetti et al., A Tutorial on Graph-Based SLAM](https://doi.org/10.1109/MITS.2010.939925)

[Dellaert and Hutchinson, SLAM](https://www.roboticsbook.org/S64_driving_perception.html)
