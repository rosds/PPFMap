#include <algorithm>

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


struct Pose {
    Eigen::Affine3f transformation;
    pcl::Correspondence corr;

    Pose(const Eigen::Affine3f& t, const pcl::Correspondence& c) 
        : transformation(t), corr(c) {}
};


inline bool posesWithinErrorBounds(
    const Eigen::Affine3f& pose1, const Eigen::Affine3f& pose2) {

    const float clustering_position_diff_threshold_ = 0.1f;
    const float clustering_rotation_diff_threshold_ = 12.0f / 180.0f * static_cast<float>(M_PI);

    // Translation difference.
    float position_diff = (pose1.translation() - pose2.translation()).norm();
    
    // Rotation angle difference.
    Eigen::AngleAxisf rotation_diff_mat(pose1.rotation().inverse() * pose2.rotation());
    float rotation_diff = fabsf(rotation_diff_mat.angle());

    return position_diff < clustering_position_diff_threshold_; // && 
           //rotation_diff < clustering_rotation_diff_threshold_;
}


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

    std::vector<Pose> pose_vector;

    pcl::CorrespondencesPtr corr(new pcl::Correspondences());

    pcl::StopWatch timer;

    timer.reset();

    int dummy = 0;
    for (size_t i = 0; i < scene_downsampled->size(); i++) {
        if (dummy % 10 == 0) {
            const auto& scene_point = scene_downsampled->at(i);

            if (!pcl::isFinite(scene_point)) continue;

            Eigen::Affine3f pose;
            int j = ppf_matching.findBestMatch(i, scene_downsampled, scene_downsampled, neighborhood_radius, pose);
            corr->push_back(pcl::Correspondence(i, j, 0.0f));

            pose_vector.push_back(Pose(pose, pcl::Correspondence(i, j, 0.0f)));
        }
        dummy++;
    }

    std::cout << "Correspondences search: " << timer.getTimeSeconds() << std::endl;

/*
 *    std::vector<std::pair<int, int> > groundtruth_vector;
 *
 *    std::ifstream groundtruth_file("output_groundtruth.txt");
 *    boost::archive::text_iarchive ia(groundtruth_file);
 *
 *    ia >> groundtruth_vector;
 *
 *    std::vector<Pose> groundtruth_poses;
 *    for (const auto& pose : pose_vector) {
 *        std::pair<int, int> match(pose.corr.index_query, pose.corr.index_match);
 *        if (std::binary_search(groundtruth_vector.begin(), groundtruth_vector.end(), match)) {
 *            groundtruth_poses.push_back(pose); 
 *        }
 *    }
 *
 *    for (const auto& pose : groundtruth_poses) {
 *        posesWithinErrorBounds(groundtruth_poses[0].transformation, pose.transformation);
 *    }
 *
 *    std::cout << "End of gt vec" << std::endl;
 */


    int cluster_idx;
    std::vector<std::pair<int, int> > cluster_votes;
    std::vector<std::vector<Pose> > pose_clusters;

    for (const auto& pose : pose_vector) {

        bool found_cluster = false;

        cluster_idx = 0;

        for (auto& cluster : pose_clusters) {

            if (posesWithinErrorBounds(pose.transformation, cluster.front().transformation)) {
                found_cluster = true;
                cluster.push_back(pose);
                cluster_votes[cluster_idx].first++;
            }
            ++cluster_idx;
        }

        // Add a new cluster of poses
        if (found_cluster == false) {
            std::vector<Pose> new_cluster;
            new_cluster.push_back(pose);
            pose_clusters.push_back(new_cluster);
            cluster_votes.push_back(std::pair<size_t, float>(1, pose_clusters.size() - 1));
        }
    }

    std::sort(cluster_votes.begin(), cluster_votes.end());

    // ========================================================================
    //  Visualize the clouds
    // ========================================================================

    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer());
    viewer->addPointCloud<pcl::PointNormal>(model_downsampled, "model_downsampled");
    viewer->addPointCloud<pcl::PointNormal>(scene_downsampled, "scene_downsampled");

    for (const auto& pose : pose_clusters[cluster_votes.back().second]) {
        sprintf(name, "line_%d_%d", pose.corr.index_query, pose.corr.index_match);    
        auto& scene_point = scene_downsampled->at(pose.corr.index_query);
        auto& model_point = model_downsampled->at(pose.corr.index_match);
        viewer->addLine(scene_point, model_point, 1.0f, 0.0f, 0.0f, name);

        sprintf(name, "sphere_%d", pose.corr.index_query);    
        //viewer->addSphere(scene_point, neighborhood_radius, name);
    }

    //viewer->setRepresentationToWireframeForAllActors();

    while (!viewer->wasStopped()) {
        viewer->spinOnce();
    }

    return 0;
}
