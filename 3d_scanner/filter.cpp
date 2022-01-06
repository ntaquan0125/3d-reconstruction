#include "filter.h"

pcl::PointCloud<pcl::PointXYZ>::Ptr voxel_grid_downsample(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float leaf_size)
{
    pcl::VoxelGrid<pcl::PointXYZ> vox;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);

    vox.setInputCloud(cloud);
    vox.setLeafSize(leaf_size, leaf_size, leaf_size);
    vox.filter(*cloud_filtered);
    return cloud_filtered;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr outlier_removal(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float threshold)
{
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);

    sor.setInputCloud(cloud);
    sor.setMeanK(50);
    sor.setStddevMulThresh(threshold);
    sor.filter(*cloud_filtered);
    return cloud_filtered;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr pass_through(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float z1, float z2 , float y1, float y2, float x1, float x2)
{
    pcl::PassThrough<pcl::PointXYZ> pass;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filteredz(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filteredy(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);

    pass.setInputCloud(cloud);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(z1, z2);
    //pass.setFilterLimitsNegative(true);
    pass.filter(*cloud_filteredz);

    pass.setInputCloud(cloud_filteredz);
    pass.setFilterFieldName("y");
    pass.setFilterLimits(y1, y2);
    pass.filter(*cloud_filteredy);

    pass.setInputCloud(cloud_filteredy);
    pass.setFilterFieldName("x");
    pass.setFilterLimits(x1, x2);
    pass.filter(*cloud_filtered);
    return cloud_filtered;
}