import copy
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from tqdm import tqdm

from registration import *
from evaluate import *


intrinsic = o3d.camera.PinholeCameraIntrinsic(
    o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
)


class Keyframe():
    def __init__(self, id, cloud, fpfh, odom):
        self.id = id
        self.cloud = copy.deepcopy(cloud)
        self.fpfh = copy.deepcopy(fpfh)
        self.odom = copy.deepcopy(odom)
        self.node = o3d.pipelines.registration.PoseGraphNode(odom)


class SLAM():
    def __init__(self, voxel_size):
        self.voxel_size = voxel_size
        self.graph = o3d.pipelines.registration.PoseGraph()
        self.keyframes = []
        self.camera_marker = []
        self.traj = []
        self.raw_traj = []

    def update(self, cloud):
        cloud = cloud.voxel_down_sample(self.voxel_size)
        cloud, fpfh = compute_features(cloud, self.voxel_size)

        if not len(self.keyframes):
            odom = np.array([[1, 0, 0, 1.5], [0, 1, 0, 1.5], [0, 0, 1, -0.3], [0, 0, 0, 1]])
            # odom = np.identity(4)
            self.keyframes.append(Keyframe(0, cloud, fpfh, odom))
            self.graph.nodes.append(self.keyframes[-1].node)

            odom_inv = np.linalg.inv(odom)
            self.frustum = o3d.geometry.LineSet.create_camera_visualization(intrinsic,odom_inv, 0.3)
            return

        if self.update_keyframe(cloud, fpfh):
            self.optimize_posegraph()


    def update_keyframe(self, cloud, fpfh):
        success, transformation, information = register_point_cloud_fpfh(
            cloud, self.keyframes[-1].cloud,
            fpfh, self.keyframes[-1].fpfh,
            self.voxel_size, True
        )

        if not success:
            return False

        odom = np.dot(self.graph.nodes[-1].pose, transformation)
        self.keyframes.append(Keyframe(len(self.keyframes), cloud, fpfh, odom))

        odom_inv = np.linalg.inv(odom)
        self.frustum = o3d.geometry.LineSet.create_camera_visualization(intrinsic, odom_inv, 0.3)
        self.camera_marker.append(self.frustum)

        self.graph.nodes.append(self.keyframes[-1].node)
        edge = o3d.pipelines.registration.PoseGraphEdge(
            self.keyframes[-1].id, self.keyframes[-2].id,
            transformation, information, uncertain=False
        )
        self.graph.edges.append(edge)

        for i in range(0, len(self.keyframes) - 5, 5):
            transformation = ominus(self.keyframes[i].odom, self.keyframes[-1].odom)
            trans = compute_distance(transformation)
            rot = compute_angle(transformation)

            if trans < 1 and rot < (np.pi / 18):
                success, transformation, information = register_point_cloud_fpfh(
                    self.keyframes[i].cloud, cloud,
                    self.keyframes[i].fpfh, fpfh,
                    self.voxel_size, False
                )
                if success:
                    edge = o3d.pipelines.registration.PoseGraphEdge(
                        self.keyframes[i].id, self.keyframes[-1].id,
                        transformation, information, uncertain=True
                    )
                    self.graph.edges.append(edge)
        return True


    def optimize_posegraph(self):
        option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=self.voxel_size * 1.2,
            edge_prune_threshold=0.25,
            reference_node=0
        )
        o3d.pipelines.registration.global_optimization(
            self.graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            option
        )


    def save_trajectory(self):
        for i in range(len(self.keyframes)):
            try:
                self.traj.append(slam.graph.nodes[i].pose)
                self.raw_traj.append(self.keyframes[i].node.pose)
            except:
                print(i)


if __name__ == '__main__':
    path = 'data/living2'
    color_dir = os.path.join(path, 'rgb')
    depth_dir = os.path.join(path, 'depth')
    traj_dir = os.path.join(path, 'traj.txt')
    num_frames = 2000
    skip_frames = 10
    slam = SLAM(0.03)
    run_times = []

    for i in range(0, num_frames, skip_frames):
        color_raw = o3d.io.read_image(color_dir + '/{0:05d}.jpg'.format(i))
        depth_raw = o3d.io.read_image(depth_dir + '/{0:05d}.png'.format(i))
        source_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_raw, depth_raw
        )

        source = o3d.geometry.PointCloud.create_from_rgbd_image(source_rgbd_image, intrinsic)

        start = time.time()
        slam.update(source)
        end = time.time()

        run_times.append(end - start)
        print(f'Step: {i}, FPS: {1 / (end - start):6f}')

    slam.save_trajectory()
    groundtruth = load_groundtruth(traj_dir, num_frames, skip_frames)

    trans_error = calculate_trans_error(groundtruth, slam.traj)
    rot_error = calculate_rot_error(groundtruth, slam.traj)
    length = distances_along_trajectory(groundtruth)
    
    print('Evaluating:')
    print(f'trajectory_length: {length:.6f}')

    print(f'translational_error.mean: {np.mean(trans_error):.6f}')
    print(f'translational_error.std: {np.std(trans_error):.6f}')

    print(f'rotational_error.mean: {np.mean(rot_error) * 180.0 / np.pi:.6f}')
    print(f'rotational_error.std: {np.std(rot_error) * 180.0 / np.pi:.6f}')

    print(f'fps.mean {1 / np.mean(run_times):.6f}')

    x1 = [t[0, 3] for t in groundtruth]
    y1 = [t[1, 3] for t in groundtruth]
    x2 = [t[0, 3] for t in slam.traj]
    y2 = [t[1, 3] for t in slam.traj]
    x3 = [t[0, 3] for t in slam.raw_traj]
    y3 = [t[1, 3] for t in slam.raw_traj]

    plt.plot(x1, y1, 'r', x2, y2, 'b', x3, y3, 'g')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('Office 2')
    plt.legend(
        ['ground truth', 'estimated trajectory', 'raw  trajectory'],
        loc ='upper right'
    )
    plt.show()

    fig, axs = plt.subplots(3, 1)
    axs[0].plot(trans_error)
    axs[1].plot(rot_error)
    axs[2].plot(run_times)

    axs[0].set_ylabel('ATE (m)')
    axs[1].set_ylabel('RE (rad)')
    axs[2].set_ylabel('run time (s)')
    axs[2].set_xlabel('step')
    plt.show()
	
    print('Generating scene:')
    final = o3d.geometry.PointCloud()
    for i in tqdm(range(len(slam.keyframes))):
        final += slam.keyframes[i].cloud.transform(slam.graph.nodes[i].pose)
        final = final.voxel_down_sample(0.001)
    o3d.visualization.draw_geometries([final] + slam.camera_marker)
    
