import copy
import time

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pyquaternion


MAX_DISTANCE = 1
MAX_ANGLE = 20
MIN_RMSE = 2.2e-2
MIN_FITNESS = 0.6


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source.paint_uniform_color([0, 1, 0])
    source_temp.paint_uniform_color([0, 0, 1])
    target_temp.paint_uniform_color([1, 0, 0])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source, source_temp, target_temp])


def preprocessing(pcd, voxel_size=0.03):
    pcd = pcd.voxel_down_sample(voxel_size)
    plane_model, inliers = pcd.segment_plane(distance_threshold=voxel_size * 1.2, ransac_n=3, num_iterations=1000)
    pcd = pcd.select_by_index(inliers, invert=True)
    _, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd = pcd.select_by_index(ind)
    return pcd


def pass_through(dic, pcd):
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    x_range = np.logical_and(points[:,0] >= dic['x'][0], points[:,0] <= dic['x'][1])
    y_range = np.logical_and(points[:,1] >= dic['y'][0], points[:,1] <= dic['y'][1])
    z_range = np.logical_and(points[:,2] >= dic['z'][0], points[:,2] <= dic['z'][1])

    pass_through_filter = np.logical_and(x_range, np.logical_and(y_range,z_range))

    pcd.points = o3d.utility.Vector3dVector(points[pass_through_filter])
    pcd.colors = o3d.utility.Vector3dVector(colors[pass_through_filter])

    return pcd


def apply_noise(pcd, mu, sigma):
    noisy_pcd = copy.deepcopy(pcd)
    points = np.asarray(noisy_pcd.points)
    points += np.random.normal(mu, sigma, size=points.shape)
    noisy_pcd.points = o3d.utility.Vector3dVector(points)
    return noisy_pcd


def compute_features(pcd, voxel_size):
    radius_normal = voxel_size * 2
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 10
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd, pcd_fpfh


def point2point_registration(source, target, distance_threshold, trans_init):
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000)
    )
    return result


def point2plane_registration(source, target, distance_threshold, trans_init): 
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000)
    )
    return result


def color_registration(source, target, distance_threshold, trans_init):
    result = o3d.pipelines.registration.registration_colored_icp(
        source, target, distance_threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationForColoredICP(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000)
    )
    return result


def sac_ia_registration(source, target, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 2
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source, target, source_fpfh, target_fpfh, True, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result


def pairwise_registration(source, target, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.2
    icp_coarse = sac_ia_registration(source, target, source_fpfh, target_fpfh, voxel_size)
    icp_fine = point2point_registration(source, target, distance_threshold, icp_coarse.transformation)
    transformation = icp_fine.transformation
    information = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, distance_threshold, icp_fine.transformation)
    return transformation, information


def register_point_cloud_fpfh(source, target, source_fpfh, target_fpfh, voxel_size, odometry=False):
    try:
        distance_threshold = voxel_size * 1.2
        icp_coarse = sac_ia_registration(source, target, source_fpfh, target_fpfh, voxel_size)

        icp_fine = point2plane_registration(source, target, distance_threshold, icp_coarse.transformation)
    
        information = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            source, target, distance_threshold, icp_fine.transformation)

        if icp_fine.transformation.trace() == 4.0:
            return (False, np.identity(4), np.zeros((6, 6)))

        angle = pyquaternion.Quaternion(matrix=icp_fine.transformation[:3, :3]).degrees
        trans = np.linalg.norm(icp_fine.transformation[:3, 3])
        if abs(trans) > MAX_DISTANCE and (abs(angle) > MAX_ANGLE or (180 - abs(angle)) > MAX_ANGLE):
            return (False, np.identity(4), np.zeros((6, 6)))

        if not odometry:
            if icp_fine.inlier_rmse > MIN_RMSE or icp_fine.fitness < MIN_FITNESS:
                return (False, np.identity(4), np.zeros((6, 6)))

            if information[5, 5] / min(len(source.points), len(target.points)) < 0.5:
                return (False, np.identity(4), np.zeros((6, 6)))
        
        return (True, icp_fine.transformation, information)
    except:
        return (False, np.identity(4), np.zeros((6, 6)))


if __name__ == '__main__':
    trans_init = np.asarray([[0.862, 0.011, -0.507, 0.05],
                         [-0.139, 0.967, -0.215, 0.03],
                         [0.487, 0.255, 0.835, 0.02],
                         [0.0, 0.0, 0.0, 1.0]])

    source = o3d.io.read_point_cloud('data/stanford_cloud/bunny.ply')
    target = copy.deepcopy(source)
    source.transform(trans_init)

    voxel_size = 0.002
    source = source.voxel_down_sample(voxel_size)
    target = target.voxel_down_sample(voxel_size)
    source = apply_noise(source, 0, 0.001)
    target = apply_noise(target, 0, 0.001)
    source, source_fpfh = compute_features(source, voxel_size)
    target, target_fpfh = compute_features(target, voxel_size)
    
    start = time.time()
    reg_p2p = point2plane_registration(source, target, voxel_size * 1.2, np.identity(4))
    print('ICP')
    print('Runtime: ', time.time() - start)
    print('Inlier Fitness: ', reg_p2p.fitness)
    print('Inlier RMSE: ', reg_p2p.inlier_rmse)

    start = time.time()
    sac_ia = sac_ia_registration(source, target, source_fpfh, target_fpfh, voxel_size)
    print('SAC-IA')
    print('Runtime: ', time.time() - start)
    print('Inlier Fitness: ', sac_ia.fitness)
    print('Inlier RMSE: ', sac_ia.inlier_rmse)

    sac_ia_icp = point2point_registration(source, target, voxel_size * 1.2, sac_ia.transformation)
    print('SAC-IA + ICP')
    print('Runtime: ', time.time() - start)
    print('Inlier Fitness: ', sac_ia_icp.fitness)
    print('Inlier RMSE: ', sac_ia_icp.inlier_rmse)

    draw_registration_result(source, target, reg_p2p.transformation)
    draw_registration_result(source, target, sac_ia.transformation)
    draw_registration_result(source, target, sac_ia_icp.transformation)
