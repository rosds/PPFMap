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

    float3 points_array[number_of_points];
    float3 normals_array[number_of_points];

    for (int i = 0; i < number_of_points; i++) {
        points_array[i] = pointToFloat3(model_->at(i));
        normals_array[i] = normalToFloat3(normals_->at(i));
    }

    model_ppf_map = cuda::setPPFMap(points_array, normals_array, 
                                    number_of_points,
                                    discretization_distance,
                                    discretization_angle);

    model_map_initialized = true;

    float diameter = model_ppf_map->getCloudDiameter();

    if (diameter * neighborhood_percentage / discretization_distance > 255.0f) {
        pcl::console::print_warn(stderr, "Warning: possible hash collitions due to distance discretization\n");
    }

    if (2.0f * static_cast<float>(M_PI) / discretization_angle > 255.0f) {
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
 *  \return True if the object appears in the scene, false otherwise.
 */
template <typename PointT, typename NormalT>
bool ppfmap::PPFMatch<PointT, NormalT>::detect(
    const PointCloudPtr cloud, 
    const NormalsPtr normals, 
    Eigen::Affine3f& trans, 
    pcl::Correspondences& correspondences) {

    float affine_s[12];
    std::vector<Pose> pose_vector;
    const float radius = model_ppf_map->getCloudDiameter() * neighborhood_percentage;

    std::vector<int> indices;
    std::vector<float> distances;

    pcl::KdTreeFLANN<PointT> kdtree;
    kdtree.setInputCloud(cloud);

    if (!use_indices) {
        ref_point_indices->resize(cloud->size());
        for (int i = 0; i < cloud->size(); i++) {
            (*ref_point_indices)[i] = i;
        }
    }

    for (const auto index : *ref_point_indices) {
        const auto& point = cloud->at(index);
        const auto& normal = normals->at(index);

        if (!pcl::isFinite(point)) continue;

        getAlignmentToX(point, normal, &affine_s);
        kdtree.radiusSearch(point, radius, indices, distances);

        auto pose = getPose(index, indices, cloud, normals, affine_s);

        if (pose.votes > 5) {
            pose_vector.push_back(getPose(index, indices, cloud, normals, affine_s));
        }
    }

    return clusterPoses(pose_vector, trans, correspondences);
}


/** \brief Perform the voting and accumulation of the PPF features in the 
 * model and returns the model index with the most votes.
 *
 *  \param[in] reference_index Index of the reference point.
 *  \param[in] indices Vector of indices of the reference point neighbors.
 *  \param[in] cloud Shared pointer to the cloud.
 *  \param[in] cloud_normals Shared pointer to the cloud normals.
 *  \param[in] affine_s Affine matrix with the rotation and translation for 
 *  the alignment of the reference point/normal with the X axis.
 *  \return The pose with the most votes in the Hough space.
 */
template <typename PointT, typename NormalT>
ppfmap::Pose ppfmap::PPFMatch<PointT, NormalT>::getPose(
    const int reference_index,
    const std::vector<int>& indices,
    const PointCloudPtr cloud,
    const NormalsPtr normals,
    const float affine_s[12]) {

    Eigen::Map<const Eigen::Matrix<float, 3, 4, Eigen::RowMajor> > Tsg_map(affine_s);

    float affine_m[12];
    std::size_t n = indices.size();

    const auto& ref_point = cloud->at(reference_index);
    const auto& ref_normal = cloud->at(reference_index);

    thrust::host_vector<uint32_t> hash_list(n);
    thrust::host_vector<float> alpha_s_list(n);

    // Compute the PPF feature for all the pairs in the neighborhood
    for (int i = 0; i < n; i++) {
        const int index = indices[i];
        const auto& point = cloud->at(index);
        const auto& normal = normals->at(index);

        // Compute the PPF between reference_point and the i-th neighbor
        hash_list[i] = computePPFFeatureHash(ref_point, ref_normal,
                                             point, normal,
                                             discretization_distance,
                                             discretization_angle);

        // Compute the alpha_s angle
        const Eigen::Vector3f transformed(Tsg_map * point.getVector4fMap());
        alpha_s_list[i] = atan2f(-transformed(2), transformed(1));
    }

    int index;
    float alpha;
    int votes;
    model_ppf_map->searchBestMatch(hash_list, alpha_s_list, index, alpha, votes);

    const auto& model_point = model_->at(index);
    const auto& model_normal = normals_->at(index);

    getAlignmentToX(model_point, model_normal, &affine_m);

    Eigen::Map<const Eigen::Matrix<float, 3, 4, Eigen::RowMajor> > Tmg_map(affine_m);

    Eigen::Affine3f Tsg, Tmg;
    Tsg.matrix().block<3, 4>(0, 0) = Tsg_map.matrix();
    Tmg.matrix().block<3, 4>(0, 0) = Tmg_map.matrix();

    // Set final pose
    Pose final_pose;
    final_pose.c = pcl::Correspondence(reference_index, index, 0.0f);
    final_pose.t = Tsg.inverse() * Eigen::AngleAxisf(alpha, Eigen::Vector3f::UnitX()) * Tmg;
    final_pose.votes = votes;
    return final_pose;
}


/** \brief True if poses are similar given the translation and rotation 
 * thresholds.
 *  \param[in] t1 First pose.
 *  \param[in] t2 Second pose.
 *  \return True if the transformations are similar
 */
template <typename PointT, typename NormalT>
bool ppfmap::PPFMatch<PointT, NormalT>::similarPoses(
    const Eigen::Affine3f& pose1, const Eigen::Affine3f& pose2) {

    // Translation difference.
    float position_diff = (pose1.translation() - pose2.translation()).norm();
    
    // Rotation angle difference.
    Eigen::AngleAxisf rotation_diff_mat(pose1.rotation().inverse() * pose2.rotation());
    float rotation_diff = fabsf(rotation_diff_mat.angle());

    return position_diff < translation_threshold &&
           rotation_diff < rotation_threshold;
}


/** \brief Returns the average pose and the correspondences for the most 
 * consistent cluster of poses.
 *  \param[in] poses Vector with the poses.
 *  \param[out] trans Average affine transformation for the biggest 
 *  cluster.
 *  \param[out] corr Vector of correspondences supporting the cluster.
 *  \return True if a cluster was found, false otherwise.
 */
template <typename PointT, typename NormalT>
bool ppfmap::PPFMatch<PointT, NormalT>::clusterPoses(
    const std::vector<Pose>& poses, 
    Eigen::Affine3f &trans, 
    pcl::Correspondences& corr) {

    if (!poses.size()) {
        return false;
    }

    int cluster_idx;
    std::vector<std::pair<int, int> > cluster_votes;
    std::vector<std::vector<Pose> > pose_clusters;

    for (const auto& pose : poses) {

        bool found_cluster = false;

        cluster_idx = 0;
        for (auto& cluster : pose_clusters) {
            if (similarPoses(pose.t, cluster.front().t)) {
                found_cluster = true;
                cluster.push_back(pose);
                cluster_votes[cluster_idx].first += pose.votes;
            }
            ++cluster_idx;
        }

        // Add a new cluster of poses
        if (found_cluster == false) {
            std::vector<Pose> new_cluster;
            new_cluster.push_back(pose);
            pose_clusters.push_back(new_cluster);
            cluster_votes.push_back(std::pair<int, int>(pose.votes , pose_clusters.size() - 1));
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

    return true;
}
