# RoboMaster Examples

Eight original hardware programs for the DJI RoboMaster. Their Python source is preserved under descriptive folder and file names. These programs use the RoboMaster SDK and some expect camera feedback, a keyboard or an attached gripper. Several initialise the hardware at module load time.

| Hardware example | Purpose |
|---|---|
| [Boundary Following with Infrared Sensing](Boundary%20Following%20with%20Infrared%20Sensing/boundary_following.py) | Uses infrared range feedback to follow a boundary. |
| [Bug1 Navigation](Bug1%20Navigation/bug1_navigation.py) | Implements the Bug1 navigation exercise with onboard sensor callbacks. |
| [Keyboard Teleoperation](Keyboard%20Teleoperation/keyboard_teleoperation.py) | Maps keyboard input to chassis movement commands. |
| [Infrared Range Control](Infrared%20Range%20Control/infrared_range_control.py) | Uses measured distance to control chassis motion. |
| [Line Detection with Visual Feedback](Line%20Detection%20with%20Visual%20Feedback/line_detection.py) | Processes the onboard line-detection callback and displays image feedback. |
| [Visual Marker Detection](Visual%20Marker%20Detection/marker_detection.py) | Reads visual marker observations from the robot camera. |
| [Pick and Place](Pick%20and%20Place/pick_and_place.py) | Coordinates perception, chassis movement and gripper actions. |
| [PID Trajectory Tracking](PID%20Trajectory%20Tracking/pid_trajectory_tracking.py) | Uses odometry feedback and a proportional-integral-derivative controller for trajectory tracking. |

The preserved files were checked for Python syntax. Physical behaviour requires the corresponding robot and peripherals. The [simulation labs](../Simulation%20Labs/README.md) run independently on a computer. [DJI RoboMaster developer guide](https://robomaster-dev.readthedocs.io/en/latest/)
