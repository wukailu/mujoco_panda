import mujoco
from mujoco import MjModel, MjData


class OldSim:
    data: MjData
    model: MjModel
    def __init__(self, model: MjModel):
        self.model = model
        self.data = MjData(model)

    def step(self):
        mujoco.mj_step(self.model, self.data)

    def forward(self):
        mujoco.mj_forward(self.model, self.data)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)