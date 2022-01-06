#include "registration.h"

void print4x4Matrix(const Eigen::Matrix4d & matrix)
{
    printf("Rotation matrix :\n");
    printf("    | %6.3f %6.3f %6.3f | \n", matrix(0, 0), matrix(0, 1), matrix(0, 2));
    printf("R = | %6.3f %6.3f %6.3f | \n", matrix(1, 0), matrix(1, 1), matrix(1, 2));
    printf("    | %6.3f %6.3f %6.3f | \n", matrix(2, 0), matrix(2, 1), matrix(2, 2));
    printf("Translation vector :\n");
    printf("t = < %6.3f, %6.3f, %6.3f >\n\n", matrix(0, 3), matrix(1, 3), matrix(2, 3));
}

pcl::SampleConsensusInitialAlignment<pcl::PointXYZ, pcl::PointXYZ, pcl::FPFHSignature33> init_alignment(
    pcl::PointCloud<pcl::PointXYZ>::Ptr source_points,
    pcl::PointCloud<pcl::PointXYZ>::Ptr target_points,
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr source_descriptors,
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr target_descriptors,
    float max_correspondence_distance,
    float min_sample_distance,
    float max_iterations
)
{
    pcl::SampleConsensusInitialAlignment<pcl::PointXYZ, pcl::PointXYZ, pcl::FPFHSignature33> sac_ia;
    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_source(new pcl::PointCloud<pcl::PointXYZ>);

    sac_ia.setInputSource(source_points);
    sac_ia.setSourceFeatures(source_descriptors);
    sac_ia.setInputTarget(target_points);
    sac_ia.setTargetFeatures(target_descriptors);
    sac_ia.setMinSampleDistance(min_sample_distance);
    sac_ia.setMaxCorrespondenceDistance(max_correspondence_distance);
    sac_ia.setMaximumIterations(max_iterations);
    sac_ia.align(*aligned_source);
    sac_ia.getCorrespondenceRandomness();
    return sac_ia;
}

pcl::IterativeClosestPointNonLinear<pcl::PointXYZ, pcl::PointXYZ> ICP_refine_alignment(
    pcl::PointCloud<pcl::PointXYZ>::Ptr source_points,
    pcl::PointCloud<pcl::PointXYZ>::Ptr target_points,
    float max_correspondence_distance,
    float outlier_rejection_threshold,
    float transformation_epsilon,
    float max_iterations
)
{
    pcl::IterativeClosestPointNonLinear<pcl::PointXYZ, pcl::PointXYZ> icp;
    pcl::PointCloud<pcl::PointXYZ>::Ptr registration_output(new pcl::PointCloud<pcl::PointXYZ>);

    icp.setInputSource(source_points);
    icp.setInputTarget(target_points);
    icp.setMaxCorrespondenceDistance(max_correspondence_distance);
    icp.setRANSACOutlierRejectionThreshold(outlier_rejection_threshold);
    icp.setTransformationEpsilon(transformation_epsilon);
    icp.setMaximumIterations(max_iterations);
    icp.align(*registration_output);

    return icp;
}

pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> GICP_refine_alignment(
    pcl::PointCloud<pcl::PointXYZ>::Ptr source_points,
    pcl::PointCloud<pcl::PointXYZ>::Ptr target_points,
    float max_correspondence_distance,
    float euclidean_fitness_epsilon,
    float transformation_epsilon,
    float max_iterations
)
{
    pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> gicp;
    pcl::PointCloud<pcl::PointXYZ>::Ptr registration_output(new pcl::PointCloud<pcl::PointXYZ>);

    gicp.setInputSource(source_points);
    gicp.setInputTarget(target_points);
    gicp.setMaxCorrespondenceDistance(max_correspondence_distance);
    gicp.setEuclideanFitnessEpsilon(euclidean_fitness_epsilon);
    gicp.setTransformationEpsilon(transformation_epsilon);
    gicp.setMaximumIterations(max_iterations);
    gicp.align(*registration_output);

    return gicp;
}

pcl::IterativeClosestPointWithNormals<pcl::PointNormal , pcl::PointNormal> ICP_normals_refine_alignment(
    pcl::PointCloud<pcl::PointNormal>::Ptr source_points,
    pcl::PointCloud<pcl::PointNormal>::Ptr target_points,
    float max_iterations
)
{
    pcl::IterativeClosestPointWithNormals<pcl::PointNormal , pcl::PointNormal> icp;
    pcl::PointCloud<pcl::PointNormal>::Ptr registration_output(new pcl::PointCloud<pcl::PointNormal>);

    icp.setInputSource(source_points);
    icp.setInputTarget(target_points);
    icp.setMaximumIterations(max_iterations);
    icp.align(*registration_output);
    return icp;
}