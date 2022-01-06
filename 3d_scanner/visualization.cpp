#include "visualization.h"

void view_pairs(
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer,
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2,
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1_al, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2_al
)
{
    int v1(0), v2(0);

    viewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1);
    viewer->setBackgroundColor(0, 0, 0, v1);
    viewer->addText("Before Alignment", 10, 10, "v1 text", v1);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> green(cloud1, 0, 255, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> red(cloud2, 255, 0, 0);
    viewer->addPointCloud(cloud1, green, "v1_target", v1);
    viewer->addPointCloud(cloud2, red, "v1_sourse", v1);

    viewer->createViewPort(0.5, 0.0, 1.0, 1.0, v2);
    viewer->setBackgroundColor(0, 0, 0, v2);
    viewer->addText("After Alignment", 10, 10, "v2 text", v2);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> green2(cloud1_al, 0, 255, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> red2(cloud2_al, 255, 0, 0);
    viewer->addPointCloud(cloud1_al, green2, "v2_target", v2);
    viewer->addPointCloud(cloud2_al, red2, "v2_sourse", v2);
}

void view_mesh(
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer,
    pcl::PolygonMesh mesh
)
{
    viewer->setBackgroundColor (0, 0, 0);
    viewer->addPolygonMesh(mesh, "meshes", 0);
}