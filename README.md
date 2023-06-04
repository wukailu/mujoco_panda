# Mujoco Panda for Deepmind Mujoco

* The original [project](https://github.com/justagist/mujoco_panda) is built on Mujoco-py which is developed by OpenAI.
* This project migrates it to new [Mujoco](https://github.com/deepmind/mujoco) by Deepmind.
* The function utils drawing points is not migrated, since I did not find the corresponding function. 

--------------------------------

# Mujoco Panda

- Franka Emika Panda Robot model definitions for Mujoco.
- MuJoCo-based robot simulator.
- Python 3 API for controlling and monitoring the simulated Franka Emika Panda Robot.
- Low-level controllers: direct position and torque control.
- Higher-level controller: task-space hybrid force motion controller (torque-based).

Robot models are in [mujoco_panda/models](mujoco_panda/models).

## Requirements

To use all functionalities of the provided library, the following dependencies have to be met.

- [mujoco](https://github.com/openai/mujoco-py) (Not required anymore)
- [Mujoco](https://github.com/deepmind/mujoco)
- numpy (`pip install numpy`)
- scipy (`pip install scipy`)
- quaternion (`pip install numpy-quaternion`)
- tkinter (`apt-get install python3-tk) (only for visualised debugging`)

## Setup Instructions

Once mujoco is correctly installed, this library can be used by sourcing the `set_env.sh` file.

```bash
source set_env.sh
```

## Usage

See `examples` for controller and API usage. See [MujocoRobot](mujoco_panda/mujoco_robot.py) and [PandaArm](mujoco_panda/panda_robot.py) for full documentation.
