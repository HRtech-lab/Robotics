# Robotics

Python examples for robot navigation, localisation, control and manipulation. The original RoboMaster programs and the CPU simulation course have their own entry points.

| Collection | Contents |
|---|---|
| [RoboMaster Examples](RoboMaster%20Examples/README.md) | Eight original programs for infrared sensing, keyboard teleoperation, visual line and marker detection, Bug1 navigation, pick and place, and PID trajectory tracking |
| [Simulation Labs](Simulation%20Labs/README.md) | Twelve worked experiments with source code, executed notebooks, reference papers, figures and numerical checks |

## Start with a question

**What happens when a doorway closes?** Compare the old and repaired routes in [D* Lite replanning](Simulation%20Labs/02%20D-Star%20Lite%20Replanning/README.md).

**How does the robot estimate where it is?** Follow measurements and odometry in [extended Kalman filter localisation](Simulation%20Labs/05%20Extended%20Kalman%20Filter%20Localisation/README.md), then see how a loop constraint changes the trajectory in [pose graph SLAM](Simulation%20Labs/06%20Pose%20Graph%20SLAM/README.md).

**Can demonstrations teach a reaching motion?** The [diffusion policy](Simulation%20Labs/11%20Diffusion%20Policies%20for%20Reaching/README.md) and [flow-matching policy](Simulation%20Labs/12%20Conditional%20Flow%20Matching%20for%20Reaching/README.md) train on generated actions and report errors on unseen starts and goals.

![Planned route and best feasible cost](Simulation%20Labs/03%20Informed%20RRT-Star/figure.svg)

## Running the simulations

```sh
python -m pip install -r requirements.txt
python "Simulation Labs/01 A-Star Grid Planning/a_star_grid_planning.py"
python -m unittest discover -s tests -v
```

Use Python 3.12 with a CPU build of PyTorch. The hardware examples require a compatible RoboMaster and its SDK. The simulation figures are produced by the examples themselves. Each notebook cites the relevant method. [MIT licence](LICENSE)
