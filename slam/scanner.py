import copy
import time

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import open3d.core as o3c

from registration import *


def load_point_clouds(voxel_size):
    dic = {'x':[-1.3, 1.3], 'y':[-1.3, 1.3], 'z':[0, 1.3]}
    pcds = []
    for i in range(0, 10):
        source = o3d.io.read_point_cloud('cloud/inputCloud{}.pcd'.format(i))
        # source.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        source = pass_through(dic, source)
        source = preprocessing(source, voxel_size)
        pcds.append(source)
    return pcds


def full_registration(pcds, voxel_size):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            pcds[source_id], source_fpfh = compute_features(pcds[source_id], voxel_size)
            pcds[target_id], target_fpfh = compute_features(pcds[target_id], voxel_size)
            transformation_icp, information_icp = pairwise_registration(
                pcds[source_id], pcds[target_id], source_fpfh, target_fpfh, voxel_size)
            if target_id == source_id + 1:  # Odometry case
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry)))
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=False))
            else:  # Loop closure case
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=True))
    return pose_graph


if __name__ == '__main__':
    print('Load data')
    voxel_size = 0.01
    pcds_down = load_point_clouds(voxel_size)
    o3d.visualization.draw_geometries(pcds_down)

    print('Full registration')
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        pose_graph = full_registration(pcds_down, voxel_size)

    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=voxel_size * 1.2,
        edge_prune_threshold=0.25,
        reference_node=0
    )
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option
    )

    print('Transform points and display')
    for point_id in range(len(pcds_down)):
        print(pose_graph.nodes[point_id].pose)
        pcds_down[point_id].transform(pose_graph.nodes[point_id].pose)

    pcds = load_point_clouds(voxel_size)
    pcd_combined = o3d.geometry.PointCloud()
    for point_id in range(len(pcds)):
        pcds[point_id].transform(pose_graph.nodes[point_id].pose)
        pcd_combined += pcds[point_id]
    pcd_combined_down = pcd_combined.voxel_down_sample(voxel_size=0.0001)
    _, ind = pcd_combined_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd_combined_down = pcd_combined_down.select_by_index(ind)
    o3d.io.write_point_cloud('multiway_registration.pcd', pcd_combined_down)
    o3d.visualization.draw_geometries([pcd_combined_down])

    alpha = 0.008
    print(f'Surface reconstruction with alpha={alpha:.3f}')
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd_combined_down, alpha)
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
