#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <PPFMap/PPFMatch.h>


int main(int argc, char *argv[]) {

    pcl::PointCloud<pcl::PointXYZ>::Ptr model(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr scene(new pcl::PointCloud<pcl::PointXYZ>());
    
    pcl::PointCloud<pcl::Normal>::Ptr model_normals(new pcl::PointCloud<pcl::Normal>());
    
    // ========================================================================
    //  Load the point clouds of the model and the scene
    // ========================================================================

    pcl::io::loadPCDFile("../clouds/milk.pcd", *model);
    Eigen::Matrix4f trans = Eigen::Matrix4f::Identity();
    trans(0,3) = -2.0f;
    pcl::transformPointCloud(*model, *model, trans);

    pcl::io::loadPCDFile("../clouds/milk_cartoon_all_small_clorox.pcd", *scene);

    // ========================================================================
    //  Compute the model's normals
    // ========================================================================
    
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud(model);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    ne.setSearchMethod(tree);
    ne.setRadiusSearch(0.03f);
    ne.compute(*model_normals);

    // ========================================================================
    //  Compute the model's normals
    // ========================================================================

    ppfmap::PPFMatch<pcl::PointXYZ, pcl::Normal> ppf_matching(12.0f / 180.0f * static_cast<float>(M_PI), 0.001f);
    ppf_matching.setModelPointCloud(model);
    ppf_matching.setModelNormals(model_normals);
    ppf_matching.initPPFSearchStruct();

    // ========================================================================
    //  Visualize the clouds
    // ========================================================================

    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer());
    viewer->addPointCloud(model, "model_cloud");
    viewer->addPointCloud(scene, "scene_cloud");

    viewer->addPointCloudNormals<pcl::PointXYZ, pcl::Normal> (model, model_normals, 10, 0.05, "normals");

    while (!viewer->wasStopped()) {
        viewer->spinOnce();
    }

    return 0;
}
