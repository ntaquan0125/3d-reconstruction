#pragma once

#include <pcl/io/io.h>
#include <pcl/point_types.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>

pcl::PointCloud<pcl::PointXYZ>::Ptr voxel_grid_downsample(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float leaf_size);
pcl::PointCloud<pcl::PointXYZ>::Ptr outlier_removal(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float threshold);
pcl::PointCloud<pcl::PointXYZ>::Ptr pass_through(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float z1, float z2 , float y1, float y2, float x1, float x2);
pcl::PointCloud<pcl::PointXYZ>::Ptr extract_plane(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float distance_thresh);
