#include "feature.h"

pcl::PointCloud<pcl::Normal>::Ptr get_normals(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float radius)
{
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);

    ne.setInputCloud(cloud);
    ne.setRadiusSearch(radius);
    ne.compute(*normals);
    // normals->size () should have the same size as the input cloud->size ()*
    return normals;
}

pcl::PointCloud<pcl::FPFHSignature33>::Ptr FPFH_descriptors(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::Normal>::Ptr normals, float radius)
{
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs(new pcl::PointCloud<pcl::FPFHSignature33>);

    fpfh.setInputCloud(cloud);
    fpfh.setInputNormals(normals);
    fpfh.setSearchMethod(tree);
    fpfh.setRadiusSearch(radius);
    fpfh.compute(*fpfhs);
    // fpfhs->size() should have the same size as the input cloud->size()*
    return fpfhs;
}
