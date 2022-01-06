#pragma once

#include <pcl/io/io.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/ia_ransac.h>

void print4x4Matrix(const Eigen::Matrix4d & matrix);

pcl::SampleConsensusInitialAlignment<pcl::PointXYZ, pcl::PointXYZ, pcl::FPFHSignature33> init_alignment(
    pcl::PointCloud<pcl::PointXYZ>::Ptr source_points,
    pcl::PointCloud<pcl::PointXYZ>::Ptr target_points,
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr source_descriptors,
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr target_descriptors,
    float max_correspondence_distance,
    float min_sample_distance,
    float max_iterations
);

pcl::IterativeClosestPointNonLinear<pcl::PointXYZ, pcl::PointXYZ> ICP_refine_alignment(
    pcl::PointCloud<pcl::PointXYZ>::Ptr source_points,
    pcl::PointCloud<pcl::PointXYZ>::Ptr target_points,
    float max_correspondence_distance,
    float outlier_rejection_threshold,
    float transformation_epsilon,
    float max_iterations
);

pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> GICP_refine_alignment(
    pcl::PointCloud<pcl::PointXYZ>::Ptr source_points,
    pcl::PointCloud<pcl::PointXYZ>::Ptr target_points,
    float max_correspondence_distance,
    float euclidean_fitness_epsilon,
    float transformation_epsilon,
    float max_iterations
);


// Point to plane registration
pcl::IterativeClosestPointWithNormals<pcl::PointNormal , pcl::PointNormal> ICP_normals_refine_alignment(
    pcl::PointCloud<pcl::PointNormal>::Ptr source_points,
    pcl::PointCloud<pcl::PointNormal>::Ptr target_points,
    float max_iterations
);