#include <pcl/io/pcd_io.h>
#include <pcl/correspondence.h>
#include <pcl/common/transforms.h>
#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <PPFMap/PPFMatch.h>


int main(int argc, char *argv[]) {
    char name[1024];

    pcl::PointCloud<pcl::PointXYZ>::Ptr model(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr scene(new pcl::PointCloud<pcl::PointXYZ>());
    
    pcl::PointCloud<pcl::Normal>::Ptr model_normals(new pcl::PointCloud<pcl::Normal>());
    pcl::PointCloud<pcl::Normal>::Ptr scene_normals(new pcl::PointCloud<pcl::Normal>());

    pcl::PointCloud<pcl::PointNormal>::Ptr model_with_normals(new pcl::PointCloud<pcl::PointNormal>());
    pcl::PointCloud<pcl::PointNormal>::Ptr scene_with_normals(new pcl::PointCloud<pcl::PointNormal>());
    
    // ========================================================================
    //  Load the point clouds of the model and the scene
    // ========================================================================

    pcl::io::loadPCDFile("../clouds/milk.pcd", *model);
    Eigen::Matrix4f trans = Eigen::Matrix4f::Identity();
    trans(0,3) = -2.0f;
    pcl::transformPointCloud(*model, *model, trans);

    pcl::io::loadPCDFile("../clouds/milk_cartoon_all_small_clorox.pcd", *scene);

    // ========================================================================
    //  Compute the model's normals and Downsample
    // ========================================================================
    
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud(model);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    ne.setSearchMethod(tree);
    ne.setRadiusSearch(0.03f);
    ne.compute(*model_normals);
    pcl::concatenateFields(*model, *model_normals, *model_with_normals);

    ne.setInputCloud(scene);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree2(new pcl::search::KdTree<pcl::PointXYZ>());
    ne.setSearchMethod(tree2);
    ne.setRadiusSearch(0.03f);
    ne.compute(*scene_normals);
    pcl::concatenateFields(*scene, *scene_normals, *scene_with_normals);

    pcl::PointCloud<pcl::PointNormal>::Ptr model_downsampled(new pcl::PointCloud<pcl::PointNormal>());
    pcl::VoxelGrid<pcl::PointNormal> sor;
    sor.setInputCloud(model_with_normals);
    sor.setLeafSize(0.01f, 0.01f, 0.01f);
    sor.filter(*model_downsampled);

    // ========================================================================
    //  Compute the model's ppfs
    // ========================================================================

    ppfmap::PPFMatch<pcl::PointNormal, pcl::PointNormal> ppf_matching(12.0f / 180.0f * static_cast<float>(M_PI), 0.001f);
    ppf_matching.setModelPointCloud(model_downsampled);
    ppf_matching.setModelNormals(model_downsampled);
    ppf_matching.initPPFSearchStruct();

    // ========================================================================
    //  Find correspondences
    // ========================================================================

    pcl::CorrespondencesPtr corr(new pcl::Correspondences());
    int dummy = 0;
    for (size_t i = 0; i < scene_with_normals->size(); i++) {
        if (dummy % 10000 == 0) {
            const auto& scene_point = scene_with_normals->at(i);

            if (!pcl::isFinite(scene_point)) continue;

            int j = ppf_matching.findBestMatch(i, scene_with_normals, scene_with_normals, 0.1f);
            corr->push_back(pcl::Correspondence(i, j, 0.0f));
        }
        dummy++;

        std::cout << i << " / " << scene_with_normals->size() << std::endl;
    }

    // ========================================================================
    //  Visualize the clouds
    // ========================================================================

    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer());
    viewer->addPointCloud(scene, "scene_cloud");

    viewer->addPointCloudNormals<pcl::PointNormal, pcl::PointNormal> (model_downsampled, model_downsampled, 10, 0.05, "normals");

    for (const auto& c : *corr) {
        sprintf(name, "line_%d_%d", c.index_query, c.index_match);    
        auto& scene_point = scene->at(c.index_query);
        auto& model_point = model_downsampled->at(c.index_match);
        viewer->addLine(scene_point, model_point, name);
    }

    while (!viewer->wasStopped()) {
        viewer->spinOnce();
    }

    return 0;
}
