# Robotics Simulation Labs

These twelve experiments run on a CPU and use generated geometry, sensor measurements or demonstrations. Each one pairs a Python script with an executed notebook, a result figure and its primary references. Small fixtures let a reader measure what the method achieved and inspect where its assumptions enter.

## Planning and estimation

Compare A* and Dijkstra against an independent shortest-path calculation, then close a doorway and repair a route with D* Lite. Informed RRT* explores a continuous room and records the best route after each iteration. The estimation lectures cover point-cloud registration, landmark-based filtering and a pose graph with an inconsistent loop closure.

## Control and learned policies

The cart and mobile robot exercises measure control effort, tracking error and clearance. The arm exercises distinguish a smooth reference trajectory from the inverse problem of reaching a Cartesian target. The two learned policies share the same demonstrator, test starts, action horizon and success threshold so that their recorded results have a common interpretation.

## Worked lectures

| Lecture | Question or experiment |
|---|---|
| [01. A* Grid Planning](01%20A-Star%20Grid%20Planning/README.md) | Find a shortest route across a four-connected occupancy grid and count how much of the grid each search expands. |
| [02. D* Lite Replanning](02%20D-Star%20Lite%20Replanning/README.md) | Repair a route after a doorway closes while the robot is already partway along its journey. |
| [03. Informed RRT*](03%20Informed%20RRT-Star/README.md) | Plan a collision-free route through a continuous two-dimensional room and track how its cost improves. |
| [04. Iterative Closest Point](04%20Iterative%20Closest%20Point/README.md) | Estimate a planar rigid transform between two point clouds when point correspondences are initially unknown. |
| [05. Extended Kalman Filter Localisation](05%20Extended%20Kalman%20Filter%20Localisation/README.md) | Fuse uncertain wheel motion with intermittent range and bearing observations of known landmarks. |
| [06. Pose Graph SLAM](06%20Pose%20Graph%20SLAM/README.md) | Correct accumulated odometry drift using relative-pose constraints from repeated places. |
| [07. Constrained Model Predictive Control](07%20Constrained%20Model%20Predictive%20Control/README.md) | Move a one-dimensional cart to a target without exceeding a bound on acceleration. |
| [08. Regulated Pure Pursuit](08%20Regulated%20Pure%20Pursuit/README.md) | Track a curved indoor path while adjusting speed for curvature, obstacle proximity and the remaining distance. |
| [09. Minimum-Jerk Trajectories](09%20Minimum-Jerk%20Trajectories/README.md) | Generate a smooth rest-to-rest motion with specified endpoint position, velocity and acceleration. |
| [10. Damped Least-Squares Inverse Kinematics](10%20Damped%20Least-Squares%20Inverse%20Kinematics/README.md) | Move a planar arm towards a reachable target using a regularised Jacobian inverse. |
| [11. Diffusion Policies for Reaching](11%20Diffusion%20Policies%20for%20Reaching/README.md) | Learn short actions for a simulated tabletop point moving towards a goal from offline demonstrations. |
| [12. Conditional Flow Matching for Reaching](12%20Conditional%20Flow%20Matching%20for%20Reaching/README.md) | Learn short actions for a simulated tabletop point moving towards a goal from offline demonstrations. |

## Reproducing a result

Install [the Python dependencies](../requirements.txt). Open a numbered notebook in its own directory or run the adjacent Python script. The learning examples train small PyTorch networks from scratch and take longer than the other labs. The RoboMaster SDK is used only by the separate hardware examples. [MIT licence](../LICENSE)
