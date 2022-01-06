#pragma once

#include <pcl/point_types.h>
#include <pcl/point_representation.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/histogram_visualizer.h>
#include <pcl/visualization/pcl_visualizer.h>

void view_pairs(
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer,
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2,
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1_al, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2_al
);

void view_mesh(
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer,
    pcl::PolygonMesh mesh
);