#include <PPFMap/PPFMatch.h>


/** \brief Construct the PPF search structures for the model cloud.
 *  
 *  The model cloud contains the information about the object that is going 
 *  to be detected in the scene cloud. The necessary information to build 
 *  the search structure are the points and normals from the object.
 *
 *  \param[in] model Point cloud containing the model object.
 *  \param[in] normals Cloud with the normals of the object.
 */
template <typename PointT, typename NormalT>
void ppfmap::PPFMatch<PointT, NormalT>::setModelCloud(
    const PointCloudPtr model, const NormalsPtr normals)  {

    model_ = model;
    normals_ = normals;

    const std::size_t number_of_points = model_->size();

    std::unique_ptr<float[]> points_array(new float[3 * number_of_points]);
    std::unique_ptr<float[]> normals_array(new float[3 * number_of_points]);

    for (int i = 0; i < number_of_points; i++) {
        const auto& point = model_->at(i); 
        const auto& normal = normals_->at(i); 

        points_array[i * 3 + 0] = point.x;
        points_array[i * 3 + 1] = point.y;
        points_array[i * 3 + 2] = point.z;

        normals_array[i * 3 + 0] = normal.normal_x;
        normals_array[i * 3 + 1] = normal.normal_y;
        normals_array[i * 3 + 2] = normal.normal_z;
    }

    model_ppf_map = cuda::setPPFMap(points_array.get(), 
                                    normals_array.get(), 
                                    number_of_points,
                                    discretization_distance,
                                    discretization_angle);

    model_map_initialized = true;

    float diameter = model_ppf_map->getCloudDiameter();

    if (diameter / discretization_distance > 256.0f) {
        pcl::console::print_warn(stderr, "Warning: possible hash collitions due to distance discretization\n");
    }

    if (2.0f * static_cast<float>(M_PI) / discretization_angle > 256.0f) {
        pcl::console::print_warn(stderr, "Warning: possible hash collitions due to angle discretization\n");
    }
}


/** \brief Search of the model in an scene cloud and returns the 
 * correspondences and the transformation to the scene.
 *
 *  \param[in] cloud Point cloud of the scene.
 *  \param[in] normals Normals of the scene cloud.
 *  \param[out] trans Affine transformation from to model to the scene.
 *  \param[out] correspondence Supporting correspondences from the scene to 
 *  the model.
 */
template <typename PointT, typename NormalT>
void ppfmap::PPFMatch<PointT, NormalT>::detect(
    const PointCloudPtr cloud, 
    const NormalsPtr normals, 
    Eigen::Affine3f& trans, 
    pcl::Correspondences& correspondences) {

    float radius = model_ppf_map->getCloudDiameter() / 2.0f;
    std::vector<Pose> pose_vector;

    int dummy = 0;
    for (std::size_t i = 0; i < cloud->size(); i++) {

        if (dummy % 50 == 0) {

        const auto& point = cloud->at(i);
        const auto& normal = normals->at(i);

        if (!pcl::isFinite(point)) continue;

        int j;
        Eigen::Affine3f pose;
        findBestMatch(i, cloud, normals, radius, j, pose);

        pose_vector.push_back(Pose(pose, pcl::Correspondence(i, j, 0.0f)));

        }
        dummy++;
    }

    clusterPoses(pose_vector, trans, correspondences);
}


template <typename PointT, typename NormalT>
int ppfmap::PPFMatch<PointT, NormalT>::findBestMatch(
    const int point_index,
    const PointCloudPtr cloud,
    const NormalsPtr cloud_normals,
    const float radius_neighborhood,
    int& m_idx,
    Eigen::Affine3f& pose) {

    float affine_s[12];
    float affine_m[12];
    float radius;

    // Check that the neighborhood is not bigger than the object itself 
    if (radius_neighborhood > model_ppf_map->getCloudDiameter()) {
        pcl::console::print_warn(stderr, "Warning: neighborhood radius bigger than the object diameter\n");
        radius = model_ppf_map->getCloudDiameter();
    } else {
        radius = radius_neighborhood;
    }

    const auto& ref_point = cloud->at(point_index);
    const auto& ref_normal = cloud_normals->at(point_index);

    getAlignmentToX(ppfmap::pointToFloat3(ref_point), 
                    ppfmap::normalToFloat3(ref_normal), 
                    &affine_s);

    std::vector<int> indices;
    std::vector<float> distances;
    pcl::KdTreeFLANN<PointT> kdtree;
    kdtree.setInputCloud(cloud);
    kdtree.radiusSearch(ref_point, radius, indices, distances);

    thrust::host_vector<uint32_t> hash_list;
    thrust::host_vector<float> alpha_s_list;

    // Compute the PPF feature for all the pairs in the neighborhood
    for (const auto index : indices) {

        const auto& point = cloud->at(index);
        const auto& normal = cloud_normals->at(index);

        // Transform the point and compute the alpha_s
        float d_y = point.x * affine_s[4] + point.y * affine_s[5] + point.z * affine_s[6] + affine_s[7]; 
        float d_z = point.x * affine_s[8] + point.y * affine_s[9] + point.z * affine_s[10] + affine_s[11]; 

        float alpha_s = atan2f(-d_z, d_y);
    
        uint32_t hash_key = computePPFFeatureHash(pointToFloat3(ref_point), 
                                                  normalToFloat3(ref_normal),
                                                  pointToFloat3(point), 
                                                  normalToFloat3(normal),
                                                  discretization_distance,
                                                  discretization_angle);

        hash_list.push_back(hash_key);
        alpha_s_list.push_back(alpha_s);
    }


    int index;
    float alpha;
    model_ppf_map->searchBestMatch(hash_list, alpha_s_list, index, alpha);

    const auto& model_point = model_->at(index);
    const auto& model_normal = normals_->at(index);

    getAlignmentToX(ppfmap::pointToFloat3(model_point), 
                    ppfmap::normalToFloat3(model_normal), 
                    &affine_m);

    Eigen::Map<Eigen::Matrix<float, 3, 4, Eigen::RowMajor> > Tsg_map(affine_s);
    Eigen::Map<Eigen::Matrix<float, 3, 4, Eigen::RowMajor> > Tmg_map(affine_m);

    Eigen::Affine3f Tsg(Eigen::Translation3f(Tsg_map.block<3, 1>(0, 3)) * Eigen::AngleAxisf(Tsg_map.block<3, 3>(0, 0)));
    Eigen::Affine3f Tmg(Eigen::Translation3f(Tmg_map.block<3, 1>(0, 3)) * Eigen::AngleAxisf(Tmg_map.block<3, 3>(0, 0)));

    pose = Tsg.inverse() * Eigen::AngleAxisf(alpha, Eigen::Vector3f::UnitX()) * Tmg;
    m_idx = index;
}


template <typename PointT, typename NormalT>
bool ppfmap::PPFMatch<PointT, NormalT>::posesWithinErrorBounds(
    const Eigen::Affine3f& pose1, const Eigen::Affine3f& pose2) {

    const float clustering_position_diff_threshold_ = 0.1f;
    const float clustering_rotation_diff_threshold_ = 12.0f / 180.0f * static_cast<float>(M_PI);

    // Translation difference.
    float position_diff = (pose1.translation() - pose2.translation()).norm();
    
    // Rotation angle difference.
    Eigen::AngleAxisf rotation_diff_mat(pose1.rotation().inverse() * pose2.rotation());
    float rotation_diff = fabsf(rotation_diff_mat.angle());

    return position_diff < clustering_position_diff_threshold_ &&
           rotation_diff < clustering_rotation_diff_threshold_;
}


template <typename PointT, typename NormalT>
void ppfmap::PPFMatch<PointT, NormalT>::clusterPoses(
    const std::vector<Pose>& poses, 
    Eigen::Affine3f &trans, 
    pcl::Correspondences& corr) {

    int cluster_idx;
    std::vector<std::pair<int, int> > cluster_votes;
    std::vector<std::vector<Pose> > pose_clusters;

    for (const auto& pose : poses) {

        bool found_cluster = false;

        cluster_idx = 0;
        for (auto& cluster : pose_clusters) {
            if (posesWithinErrorBounds(pose.t, cluster.front().t)) {
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
            cluster_votes.push_back(std::pair<int, int>(1, pose_clusters.size() - 1));
        }
    }

    std::sort(cluster_votes.begin(), cluster_votes.end());
    
    Eigen::Vector3f translation_average (0.0, 0.0, 0.0);
    Eigen::Vector4f rotation_average (0.0, 0.0, 0.0, 0.0);

    for (const auto& pose : pose_clusters[cluster_votes.back().second]) {
        translation_average += pose.t.translation();
        rotation_average += Eigen::Quaternionf(pose.t.rotation()).coeffs();
        corr.push_back(pose.c);
    }

    translation_average /= static_cast<float> (pose_clusters[cluster_votes.back().second].size());
    rotation_average /= static_cast<float> (pose_clusters[cluster_votes.back().second].size());

    trans.translation().matrix() = translation_average;
    trans.linear().matrix() = Eigen::Quaternionf(rotation_average).normalized().toRotationMatrix();
}
