import os
import time

import mujoco
import numpy as np

from mujoco_panda import PandaArm
from mujoco_panda.controllers.torque_based_controllers import OSHybridForceMotionController
from mujoco_panda.utils.viewer_utils import render_frame
from utils.debug_utils import ParallelPythonCmd


def exec_func(cmd):
    if cmd == '':
        return None
    a = eval(cmd)
    print(cmd)
    print(a)
    if a is not None:
        return str(a)

MODEL_PATH = os.environ['MJ_PANDA_PATH'] + \
    '/mujoco_panda/models/'


# controller parameters                                                     # PD control
KP_P = np.array([7000., 7000., 7000.])
KP_O = np.array([3000., 3000., 3000.])
ctrl_config = {
    # --------------    P control   ---------------
    'kp_p': KP_P,                                                           # for delta position
    'kd_p': 2.*np.sqrt(KP_P),                                               # for delta velocity
    'kp_o': KP_O,  # 10gains for orientation                                # for delta orientation
    'kd_o': [1., 1., 1.],  # gains for orientation                          # for delta omega (角速度)
    # --------------    D control   ---------------
    'ft_dir': [0, 0, 0, 0, 0, 0],                                           # enabling signal for force control [pos, torque]
    'kp_f': [1., 1., 1.],  # gains for force                                # for delta force
    'kd_f': [0., 0., 0.],  # 25gains for force                              # for delta velocity
    'kp_t': [1., 1., 1.],  # gains for torque                               # for delta torque
    'kd_t': [0., 0., 0.],  # gains for torque                               # for delta omega
    'alpha': 3.14*0,                                                        # not used anywhere
    # --------------    null control --------------
    'use_null_space_control': True,                                         # null space control 这里设定的null space control的次要任务是使得机械臂关键尽可能保持本身的位置。
    # newton meter
    'null_kp': np.array([1,2,1,2,1,2,1]) * 200,                              # weight for null space task
    'null_kd': 0,                                                           # not used anywhere
    'null_ctrl_wt': 2.5,                                                    # not used anywhere
    'use_orientation_ctrl': True,                                           # whether to control orientation
    'linear_error_thr': 0.025,                                              # linear error threshold, less than this will be regard as 0
    'angular_error_thr': 0.01,                                              # angular error threshold, less than this will be regard as 0
}

def compute_force_on_points(pt):
    global kdtree
    # Define the query point
    query = pt
    # Find the nearest point in the point cloud
    [k, idx, _] = kdtree.search_knn_vector_3d(query, 1)
    # Get the coordinates of the nearest point
    nearest = np.asarray(pcd.points)[idx[0]]
    distance = np.linalg.norm(nearest - pt)
    # Print the result
    return 0.1 / (distance**2) * ((pt - nearest) / distance), distance

def null_space_avoidance(arm: PandaArm):
    sum_u = [0. ] * 7
    min_dist = 1000
    min_id = -1
    id_force = 0
    for id in range(7):
        body_pose = arm.body_pose(f"panda_link{id+1}")[0]
        force, dist = compute_force_on_points(body_pose)
        if dist < min_dist:
            min_id = id
            min_dist = dist
            id_force = force

    jaco = arm.body_jacobian(f"panda_link{min_id+1}", list(range(min_id+1)))
    f = np.hstack([id_force, (0., 0., 0.)])    # (6,)
    u = np.dot(jaco.T, f)
    sum_u[:min_id+1] = sum_u[:min_id+1] + u
    # return sum_u + (arm._neutral_pose - arm.joint_positions()[:7]) * 0.4
    return sum_u

target_traj = []        # [(target_pos, target_ori),...]

def run():
    global target_traj
    robot_pos, robot_ori = 0, 0
    for i in range(len(target_traj)):
        # print(i, "/ ", len(target_traj))
        elapsed_r = 0.0
        now_r = time.time()
        while elapsed_r < 0.1:
            elapsed_r = time.time() - now_r

            # get current robot end-effector pose
            robot_pos, robot_ori = p.ee_pose()

            # render controller target and current ee pose using frames
            render_frame(p.viewer, robot_pos, robot_ori)
            render_frame(p.viewer, target_traj[i][0], target_traj[i][1], alpha=0.2)

            ctrl.set_goal(target_traj[i][0], target_traj[i][1])

            ctrl.update_and_step()
            p.render() # render the visualisation

    print(f"final errors: ", np.linalg.norm(target_traj[-1][0] - robot_pos),
          np.linalg.norm(quatdiff_in_euler(target_traj[-1][1], robot_ori)))
    target_traj = [target_traj[-1]]
def move_to(target_xyz, target_ori=None):
    global target_traj
    if len(target_traj) == 0:
        assert target_ori is not None
        target_traj = [(target_xyz, target_ori)]
        return
    if target_ori is None:
        target_ori = target_traj[-1][1]

    last_xyz, last_ori = target_traj[-1]
    dist = np.linalg.norm(last_xyz - target_xyz)
    dist_ori = np.linalg.norm(quatdiff_in_euler(last_ori, target_ori))
    avg_dis = 5e-3
    steps = int((dist + dist_ori) / avg_dis)
    target_traj += [(pos, target_ori) for pos, ori in
                    zip(np.linspace(last_xyz, target_xyz, steps), np.linspace(last_ori, target_ori, steps))]
    run()

def wait(steps):
    global target_traj
    assert len(target_traj) > 0
    target_traj += [target_traj[-1]] * steps
    run()

def rotate_around(center, radius):
    center = np.array(center)
    st_pos = center + np.array([0., -radius, 0.])
    # target_traj.append((st_pos, rotate_from_ori([90, 0, 0], euler2quat([135, 0, 180]))))
    target_traj.append((st_pos, rotate_from_ori([0, 0, 0], euler2quat([135, 0, 180]))))
    wait(100)

    import math
    for phi in np.linspace(-0.5 * math.pi, 1.5*math.pi, 400):
        pos = center + np.array([np.cos(phi), np.sin(phi), 0.]) * radius
        # ori = rotate_from_ori([-phi/math.pi*180, 0, 0], euler2quat([135, 0, 180]))
        ori = euler2quat([135, 0, 180])
        target_traj.append((pos, ori))
    run()

def rotate_from_ori(euler_angle_in_degree, ori_quat):
    import utils.tf as tf
    rot_quat = tf.euler2quat(np.array(euler_angle_in_degree))
    final_quat = tf.mulQuat(rot_quat[::-1], ori_quat[::-1])[::-1]
    return final_quat   # return the quat in (i,j,k,w)

if __name__ == "__main__":
    import open3d as o3d
    import numpy as np
    # Read the mesh file
    mesh = o3d.io.read_triangle_mesh(os.path.join(MODEL_PATH, "meshes/collision/bottle.stl"))
    # Sample a point cloud from the mesh
    pcd = mesh.sample_points_uniformly(number_of_points=1000)
    pcd.points = o3d.utility.Vector3dVector(np.array(pcd.points) + np.array([0.35, 0.2, 0.4125]))
    # Create a KDTree for the point cloud
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    # init arms
    p = PandaArm(model_path=MODEL_PATH+'panda_bottle.xml',
                 render=True, compensate_gravity=False, smooth_ft_sensor=True)

    if mujoco.mj_isPyramidal(p.model):
        print("Type of friction cone is pyramidal")
    else:
        print("Type of friction cone is eliptical")

    p.set_neutral_pose()    # cur_ee (0.3, 0, 0.6) (0, -0.924, -0.382, 0)
    p.step()

    from utils.tf import quat2euler, euler2quat, quatdiff_in_euler
    from math import pi
    print("quat2euler:", quat2euler(np.array((0, -0.924, -0.382, 0))))  # [0.75 pi, 0, pi]
    print("euler2quat:", euler2quat(np.array([135, 0., 180])))
    print("double: ", euler2quat(quat2euler((0, -0.924, -0.382, 0))))

    # create controller instance with default controller gains
    ctrl = OSHybridForceMotionController(p, config=ctrl_config)
    ctrl.set_active(True) # activate controller (simulation step and controller thread now running)

    # --- define trajectory in position -----
    target_traj = []
    curr_ee, curr_ori = p.ee_pose()
    ctrl.set_gripper_actuator([-0.5, -0.5])
    # ctrl.rot_cmd_mask = np.array([0., 0., 0., 0., 0., 5., 20.])
    move_to(curr_ee, curr_ori)
    # move_to(curr_ee + [0., 0., 0.1], curr_ori)
    curr_ee, curr_ori = p.ee_pose()
    wait(50)
    move_to([0.65, 0.2, curr_ee[2]], curr_ori)
    # rotate_around([0.35, 0.2, curr_ee[2]], 0.2)
    wait(50)
    ctrl.null_space_func = null_space_avoidance
    wait(200)

    cmd = ParallelPythonCmd(exec_func)
    # --------------------------------------
    
    input("Trajectory complete. Deactivate controller")
    ctrl.set_active(False)
    ctrl.stop_controller_cleanly()
    p.viewer.close()
