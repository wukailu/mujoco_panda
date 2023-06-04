import threading
import time

import numpy as np

from mujoco_panda import PandaArm
from mujoco_panda.utils.tf import quatdiff_in_euler
from mujoco_panda.utils.viewer_utils import render_frame

"""
Simplified demo of task-space control using joint torque actuation.

Robot moves its end-effector 10cm upwards (+ve Z axis) from starting position.
"""

# --------- Modify as required ------------
# Task-space controller parameters
# stiffness gains
P_pos = 1500.
P_ori = 200.
# damping gains
D_pos = 2.*np.sqrt(P_pos)
D_ori = 1.

def force_init():
    global P_pos, P_ori, D_pos, D_ori
    # stiffness gains
    P_pos = 1500.
    P_ori = 200.
    # damping gains
    D_pos = 2.*np.sqrt(P_pos)
    D_ori = 1.

def force_decay():
    global P_pos, P_ori, D_pos, D_ori
    # stiffness gains
    P_pos *= 0.8
    P_ori *= 0.8
    # damping gains
    D_pos *= 0.8
    D_ori *= 0.8

    if max(P_pos, P_ori, D_pos, D_ori) <= 1e-10:
        force_init()
# -----------------------------------------


def compute_ts_force(curr_pos, curr_ori, goal_pos, goal_ori, curr_vel, curr_omg):
    delta_pos = (goal_pos - curr_pos).reshape([3, 1])
    delta_ori = quatdiff_in_euler(curr_ori, goal_ori).reshape([3, 1])
    # print
    F = np.vstack([P_pos*(delta_pos), P_ori*(delta_ori)]) - \
        np.vstack([D_pos*(curr_vel).reshape([3, 1]),
                   D_ori*(curr_omg).reshape([3, 1])])

    error = np.linalg.norm(delta_pos) + np.linalg.norm(delta_ori)

    return F, error


threshold = 0.005
def controller_thread(ctrl_rate):
    global p, target_pos, error


    target_pos = curr_ee.copy()
    if run_controller:
        if error > threshold:
            now_c = time.time()
            curr_pos, curr_ori = p.ee_pose()
            curr_vel, curr_omg = p.ee_velocity()

            target_pos[2] = z_target
            F, new_error = compute_ts_force(
                curr_pos, curr_ori, target_pos, original_ori, curr_vel, curr_omg)

            if new_error > error:
                force_decay()

            error = new_error
            if error <= threshold:
                return

            impedance_acc_des = np.dot(p.jacobian().T, F).flatten().tolist()

            p.set_joint_commands(impedance_acc_des, compensate_dynamics=True)

            p.step(render=False)

            elapsed_c = time.time() - now_c
            sleep_time_c = (1./ctrl_rate) - elapsed_c
            if sleep_time_c > 0.0:
                time.sleep(sleep_time_c)


if __name__ == "__main__":

    p = PandaArm.withTorqueActuators(render=True, compensate_gravity=True)

    ctrl_rate = 1/p.model.opt.timestep

    render_rate = 100

    p.set_neutral_pose()
    p.step()
    time.sleep(0.01)

    new_pose = p._sim.data.qpos.copy()[:7]

    curr_ee, original_ori = p.ee_pose()

    target_z_traj: list = np.linspace(curr_ee[2], curr_ee[2]+0.1, 25).tolist()
    z_target = curr_ee[2]

    target_pos = curr_ee
    run_controller = True

    error = 100.
    # ctrl_thread = threading.Thread(target=controller_thread, args=[ctrl_rate])
    # ctrl_thread.start()

    now_r = time.time()
    i = 0
    for z_target in target_z_traj:
        print("now target: ", z_target)
        error = 100.
        while error > threshold:
            print("error now:", error, " threshold: ", threshold, "P_pos:", P_pos)
            robot_pos, robot_ori = p.ee_pose()
            controller_thread(ctrl_rate)

            render_frame(p.viewer, robot_pos, robot_ori)
            render_frame(p.viewer, target_pos, original_ori, alpha=0.2)
            p.render()

    print("Done controlling. Press Ctrl+C to quit.")

    while True:
        robot_pos, robot_ori = p.ee_pose()
        render_frame(p.viewer, robot_pos, robot_ori)
        render_frame(p.viewer, target_pos, original_ori, alpha=0.2)
        p.render()
