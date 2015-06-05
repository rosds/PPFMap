#include <algorithm>

#include <pcl/common/geometry.h>
#include <pcl/io/pcd_io.h>
#include <pcl/correspondence.h>
#include <pcl/common/time.h>
#include <pcl/common/transforms.h>
#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/visualization/pcl_visualizer.h>

/*
 *#include <boost/archive/text_iarchive.hpp>
 *#include <boost/serialization/vector.hpp>
 *#include <boost/serialization/utility.hpp>
 */

#include <PPFMap/PPFMatch.h>


int main(int argc, char *argv[]) {
    char name[1024];
    const float neighborhood_radius = 0.1f;

    pcl::PointCloud<pcl::PointXYZ>::Ptr model(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr scene(new pcl::PointCloud<pcl::PointXYZ>());
    
    pcl::PointCloud<pcl::Normal>::Ptr model_normals(new pcl::PointCloud<pcl::Normal>());
    pcl::PointCloud<pcl::Normal>::Ptr scene_normals(new pcl::PointCloud<pcl::Normal>());

    pcl::PointCloud<pcl::PointNormal>::Ptr model_with_normals(new pcl::PointCloud<pcl::PointNormal>());
    pcl::PointCloud<pcl::PointNormal>::Ptr scene_with_normals(new pcl::PointCloud<pcl::PointNormal>());

    pcl::PointCloud<pcl::PointNormal>::Ptr scene_downsampled(new pcl::PointCloud<pcl::PointNormal>());
    pcl::PointCloud<pcl::PointNormal>::Ptr model_downsampled(new pcl::PointCloud<pcl::PointNormal>());

    // ========================================================================
    //  Load the point clouds of the model and the scene
    // ========================================================================

    pcl::io::loadPCDFile("../clouds/milk.pcd", *model);
    pcl::io::loadPCDFile("../clouds/milk_cartoon_all_small_clorox.pcd", *scene);

    /*
     *pcl::io::loadPCDFile("../clouds/model_chair.pcd", *model_downsampled);
     *pcl::io::loadPCDFile("../clouds/scene_chair.pcd", *scene_downsampled);
     */

    // ========================================================================
    //  Compute normals
    // ========================================================================
    
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud(model);
    ne.setSearchMethod(tree);
    ne.setRadiusSearch(0.03f);
    ne.compute(*model_normals);
    pcl::concatenateFields(*model, *model_normals, *model_with_normals);

    ne.setInputCloud(scene);
    ne.setSearchMethod(tree);
    ne.setRadiusSearch(0.03f);
    ne.compute(*scene_normals);
    pcl::concatenateFields(*scene, *scene_normals, *scene_with_normals);

    // ========================================================================
    //  Downsample the clouds
    // ========================================================================
    
    pcl::VoxelGrid<pcl::PointNormal> sor;
    sor.setInputCloud(model_with_normals);
    sor.setLeafSize(0.01f, 0.01f, 0.01f);
    sor.filter(*model_downsampled);

    sor.setInputCloud(scene_with_normals);
    sor.setLeafSize(0.01f, 0.01f, 0.01f);
    sor.filter(*scene_downsampled);


    Eigen::Matrix4f trans = Eigen::Matrix4f::Identity();
    trans(0,3) = -4.0f;
    pcl::transformPointCloudWithNormals(*model_downsampled, *model_downsampled, trans);

    // ========================================================================
    //  Compute the model's ppfs
    // ========================================================================

    ppfmap::PPFMatch<pcl::PointNormal, pcl::PointNormal> ppf_matching(24.0f / 180.0f * static_cast<float>(M_PI), 0.01f);
    ppf_matching.setModelCloud(model_downsampled, model_downsampled);

    // ========================================================================
    //  Find correspondences
    // ========================================================================

    pcl::StopWatch timer;

    timer.reset();

    Eigen::Affine3f T;
    pcl::CorrespondencesPtr corr(new pcl::Correspondences());
    ppf_matching.detect(scene_downsampled, scene_downsampled, T, *corr);

    std::cout << "Object detection: " << timer.getTimeSeconds() << "s" <<  std::endl;

    // ========================================================================
    //  Visualize the clouds
    // ========================================================================

    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer());
    viewer->addPointCloud<pcl::PointNormal>(model_downsampled, "model_downsampled");
    viewer->addPointCloud<pcl::PointNormal>(scene_downsampled, "scene_downsampled");

    for (const auto& c : *corr) {
        auto& scene_point = scene_downsampled->at(c.index_query);
        auto& model_point = model_downsampled->at(c.index_match);

        sprintf(name, "line_%d_%d", c.index_query, c.index_match);    
        viewer->addLine(scene_point, model_point, 1.0f, 0.0f, 0.0f, name);
    }

    pcl::PointCloud<pcl::PointNormal>::Ptr model_transformed(new pcl::PointCloud<pcl::PointNormal>());

    pcl::transformPointCloud(*model_downsampled, *model_transformed, T);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointNormal> green(model_transformed, 0, 255, 0);
    viewer->addPointCloud<pcl::PointNormal>(model_transformed, green, "model_transformed");

    while (!viewer->wasStopped()) {
        viewer->spinOnce();
    }

    return 0;
}
