#include <iostream>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/point_types.h>
#include <pcl/console/time.h>

#include "feature.h"
#include "filter.h"
#include "meshing.h"
#include "registration.h"
#include "visualization.h"

int main(int argc, char **argv)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr source(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr target(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr source_down(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr target_down(new pcl::PointCloud<pcl::PointXYZ>);

    if (argc < 3)
    {
        PCL_ERROR ("Usage: ./main source.pcd target.pcd\n");
        return (-1);
    }
    pcl::io::loadPCDFile(argv[1], *source);
    pcl::io::loadPCDFile(argv[2], *target);

    source_down = pass_through(source, 0, 1.2, -1.5, 1.5, -1.5, 1.5);
    target_down = pass_through(target, 0, 1.2, -1.5, 1.5, -1.5, 1.5);

    source_down = voxel_grid_downsample(source_down, 0.01);
    target_down = voxel_grid_downsample(target_down, 0.01);

    source_down = extract_plane(source_down, 0.02);
    target_down = extract_plane(target_down, 0.02);

    source_down = outlier_removal(source_down, 2);
    target_down = outlier_removal(target_down, 2);

    pcl::PointCloud<pcl::Normal>::Ptr src_normal = get_normals(source_down, 0.1);
    pcl::PointCloud<pcl::Normal>::Ptr tgt_normal = get_normals(target_down, 0.1);

    pcl::PointCloud<pcl::FPFHSignature33>::Ptr src_features = FPFH_descriptors(source_down, src_normal, 0.25);
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr tgt_features = FPFH_descriptors(target_down, tgt_normal, 0.25);

    pcl::console::TicToc time;
    time.tic();

    Eigen::Matrix4f initial_alignment_matrix = Eigen::Matrix4f::Identity();
    Eigen::Matrix4d transformation_matrix = Eigen::Matrix4d::Identity();
    pcl::SampleConsensusInitialAlignment<pcl::PointXYZ, pcl::PointXYZ, pcl::FPFHSignature33> sac_ia = init_alignment(
        source_down, target_down, src_features, tgt_features, 20, 0.05, 2000);

    initial_alignment_matrix = sac_ia.getFinalTransformation();
    pcl::transformPointCloud(*source_down, *source_down, initial_alignment_matrix);

    pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp = GICP_refine_alignment(
        source_down, target_down, 0.1, 0.01, 1e-6, 2000);

    if (icp.hasConverged())
    {
        std::cout << "\nICP has converged, score is " << icp.getFitnessScore() << std::endl;
        transformation_matrix = icp.getFinalTransformation().cast<double>();
        print4x4Matrix(transformation_matrix);
        std::cout << "Time: " << time.toc() << " ms" << std::endl;
    }
    else
    {
        PCL_ERROR("ICP has not converged.\n");
        return (-1);
    }

    pcl::transformPointCloud(*source_down, *source_down, transformation_matrix);

    transformation_matrix = initial_alignment_matrix.cast<double>() * transformation_matrix.cast<double>();

    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->initCameraParameters();
    viewer->addCoordinateSystem(1.0);

    view_pairs(viewer, source, target, source_down, target_down);

    // *source_down += *target_down;
    // pcl::PointCloud<pcl::PointXYZ>::Ptr mls_cloud = smoothing(source_down);
    // pcl::PolygonMesh mesh = grid_projection(mls_cloud, get_normals(mls_cloud, 0.25));

    // view_mesh(viewer, mesh);

    while (!viewer->wasStopped())
    {
        viewer->spinOnce(100);
        boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    }

    return (0);
}