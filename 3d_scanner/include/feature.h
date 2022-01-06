#pragma once

#include <pcl/io/io.h>
#include <pcl/point_types.h>
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/fpfh.h>

pcl::PointCloud<pcl::Normal>::Ptr get_normals(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float radius);
pcl::PointCloud<pcl::FPFHSignature33>::Ptr FPFH_descriptors(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::Normal>::Ptr normals, float radius);
