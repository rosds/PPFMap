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

    if (diameter / discretization_distance > 255.0f) {
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
 */
template <typename PointT, typename NormalT>
void ppfmap::PPFMatch<PointT, NormalT>::detect(
    const PointCloudPtr cloud, 
    const NormalsPtr normals, 
    Eigen::Affine3f& trans, 
    pcl::Correspondences& correspondences) {

    float affine_s[12];
    std::vector<Pose> pose_vector;
    const float radius = model_ppf_map->getCloudDiameter() * 0.6f;

    std::vector<int> indices;
    std::vector<float> distances;

    pcl::KdTreeFLANN<PointT> kdtree;
    kdtree.setInputCloud(cloud);

    int dummy = 0;
    for (std::size_t i = 0; i < cloud->size(); i++) {

        if (dummy % 10 == 0) {

        const auto& point = cloud->at(i);
        const auto& normal = normals->at(i);

        if (!pcl::isFinite(point)) continue;

        getAlignmentToX(ppfmap::pointToFloat3(point), 
                        ppfmap::normalToFloat3(normal), 
                        &affine_s);

        kdtree.radiusSearch(point, radius, indices, distances);

        Pose pose;
        getPose(i, indices, cloud, normals, affine_s, &pose);

        pose_vector.push_back(pose);
        //pose_vector.push_back(Pose(pose, pcl::Correspondence(i, j, 0.0f), votes));

        }
        dummy++;
    }

    clusterPoses(pose_vector, trans, correspondences);
}


/** \brief Perform the voting and accumulation of the PPF features in the 
 * model and returns the model index with the most votes.
 *
 *  \param[in] reference_index Index of the reference point.
 *  \param[in] cloud_normals The pointer to the normals of the cloud.
 *  \param[in] neighborhood_radius The radius to consider for building 
 *  pairs around the reference point.
 *  \param[out] final_pose Resulting pose after the Hough voting.
 *  \return The index of the model point with the higher number of votes.
 */
template <typename PointT, typename NormalT>
int ppfmap::PPFMatch<PointT, NormalT>::getPose(
    const int reference_index,
    const std::vector<int>& indices,
    const PointCloudPtr cloud,
    const NormalsPtr normals,
    const float affine_s[12],
    Pose* final_pose) {

    float affine_m[12];
    std::size_t n = indices.size();

    const float3 ref_point = ppfmap::pointToFloat3(cloud->at(reference_index));
    const float3 ref_normal = ppfmap::normalToFloat3(cloud->at(reference_index));

    thrust::host_vector<uint32_t> hash_list(n);
    thrust::host_vector<float> alpha_s_list(n);

    // Compute the PPF feature for all the pairs in the neighborhood
    for (int i = 0; i < n; i++) {
        const int index = indices[i];

        const float3& point = ppfmap::pointToFloat3(cloud->at(index));
        const float3& normal = ppfmap::normalToFloat3(normals->at(index));

        // Compute the PPF between reference_point and the i-th neighbor
        hash_list[i] = computePPFFeatureHash(ref_point, ref_normal,
                                             point, normal,
                                             discretization_distance,
                                             discretization_angle);

        // Consider only the y and z plane for alpha_s calculation
        float d_y = point.x * affine_s[4] + point.y * affine_s[5] + point.z * affine_s[6] + affine_s[7]; 
        float d_z = point.x * affine_s[8] + point.y * affine_s[9] + point.z * affine_s[10] + affine_s[11]; 
        alpha_s_list[i] = atan2f(-d_z, d_y);
    }

    int index;
    float alpha;
    int votes;
    model_ppf_map->searchBestMatch(hash_list, alpha_s_list, index, alpha, votes);

    const auto& model_point = model_->at(index);
    const auto& model_normal = normals_->at(index);

    getAlignmentToX(ppfmap::pointToFloat3(model_point), 
                    ppfmap::normalToFloat3(model_normal), 
                    &affine_m);

    Eigen::Map<const Eigen::Matrix<float, 3, 4, Eigen::RowMajor> > Tsg_map(affine_s);
    Eigen::Map<const Eigen::Matrix<float, 3, 4, Eigen::RowMajor> > Tmg_map(affine_m);

    Eigen::Affine3f Tsg(Eigen::Translation3f(Tsg_map.block<3, 1>(0, 3)) * Eigen::AngleAxisf(Tsg_map.block<3, 3>(0, 0)));
    Eigen::Affine3f Tmg(Eigen::Translation3f(Tmg_map.block<3, 1>(0, 3)) * Eigen::AngleAxisf(Tmg_map.block<3, 3>(0, 0)));

    final_pose->c = pcl::Correspondence(reference_index, index, 0.0f);
    final_pose->t = Tsg.inverse() * Eigen::AngleAxisf(alpha, Eigen::Vector3f::UnitX()) * Tmg;
    final_pose->votes = votes;
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
            if (similarPoses(pose.t, cluster.front().t)) {
                found_cluster = true;
                cluster.push_back(pose);
                cluster_votes[cluster_idx].first++; //= pose.votes;
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
