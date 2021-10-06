#include <iostream>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/point_types.h>

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: ./main file.pcd" << std::endl;
        return (-1);
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ> (argv[1], *cloud) == -1)
    {
        PCL_ERROR ("Couldn't read file\n");
        return (-1);
    }

    // Populate cloud
    pcl::visualization::CloudViewer viewer ("Simple Cloud Viewer");
    viewer.showCloud (cloud);
    while (!viewer.wasStopped ())
    {
    }

    return 0;
}
