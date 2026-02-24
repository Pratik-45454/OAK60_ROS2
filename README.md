# OAK60_ROS2
# ROS2 OAK + AK60

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

ROS2 workspace integrating **OAK camera** input with **AK60 actuator** control. Supports velocity commands, serial safety, and actuator feedback.

## Install & Build
```bash
git clone git@github.com:Robotics-Club-Pulchowk/OAK60_ROS2.git
cd OAK60_ROS2
colcon build
source install/setup.bash