#include <PPFMap/CudaPPFMatch.h>


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
void ppfmap::CudaPPFMatch<PointT, NormalT>::setModelCloud(
    const PointCloudPtr& model, const NormalsPtr& normals)  {

    // Keep a reference to the point clouds
    model_ = model;
    normals_ = normals;

    const std::size_t number_of_points = model_->size();

    float3 points_array[number_of_points];
    float3 normals_array[number_of_points];

    // Copy the cloud information into float3 arrays
    for (int i = 0; i < number_of_points; i++) {
        points_array[i] = pointToFloat3(model_->at(i));
        normals_array[i] = normalToFloat3(normals_->at(i));
    }

    map = cuda::setPPFMap(points_array, normals_array, 
                          number_of_points, 
                          distance_step, angle_step);

    model_map_initialized = true;
}


/** \brief Search of the model in an scene cloud and returns the 
 * correspondences and the transformation to the scene.
 *
 *  \param[in] cloud Point cloud of the scene.
 *  \param[in] normals Normals of the scene cloud.
 *  \param[out] trans Affine transformation from to model to the scene.
 *  \param[out] correspondence Supporting correspondences from the scene to 
 *  the model.
 *  \param[out] Number of votes supporting the final pose.
 *  \return True if the object appears in the scene, false otherwise.
 */
template <typename PointT, typename NormalT>
void ppfmap::CudaPPFMatch<PointT, NormalT>::detect(
    const PointCloudPtr cloud, 
    const NormalsPtr normals, 
    Eigen::Affine3f& trans, 
    pcl::Correspondences& correspondences,
    int& votes) {

    std::vector<Pose> poses;
    detect(cloud, normals, poses);
    clusterPoses(
        poses,
        translation_threshold,
        rotation_threshold,
        trans, 
        correspondences,
        votes);
}


/** \brief Search the given scene for the object and returns a vector with 
 * the poses sorted by the votes obtained in the Hough space.
 *  
 *  \param[in] cloud Pointer to the scene cloud where to look for the 
 *  object.
 *  \param[in] normals Pointer to the cloud containing the scene normals.
 */
template <typename PointT, typename NormalT>
void ppfmap::CudaPPFMatch<PointT, NormalT>::detect(
    const PointCloudPtr cloud, 
    const NormalsPtr normals, 
    std::vector<Pose>& poses) {

    float affine_s[12];
    std::vector<Pose> pose_vector;
    const float radius = map->getCloudDiameter() * neighborhood_percentage;

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

    poses.clear();
    for (const auto index : *ref_point_indices) {
        const auto& point = cloud->at(index);
        const auto& normal = normals->at(index);

        if (!pcl::isFinite(point)) continue;

        getAlignmentToX(point, normal, &affine_s);
        kdtree.radiusSearch(point, radius, indices, distances);

        poses.push_back(getPose(index, indices, cloud, normals, affine_s));
    }

    sort(poses.begin(), poses.end(), 
         [](const Pose& a, const Pose& b) -> bool { 
             return a.votes > b.votes; 
         });
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
ppfmap::Pose ppfmap::CudaPPFMatch<PointT, NormalT>::getPose(
    const int reference_index,
    const std::vector<int>& indices,
    const PointCloudPtr cloud,
    const NormalsPtr normals,
    const float affine_s[12]) {

    Eigen::Map<const Eigen::Matrix<float, 3, 4, Eigen::RowMajor> > Tsg_map(affine_s);

    float affine_m[12];
    const std::size_t n = indices.size();

    const auto& ref_point = cloud->at(reference_index);
    const auto& ref_normal = normals->at(reference_index);

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
                                             distance_step,
                                             angle_step);

        // Compute the alpha_s angle
        const Eigen::Vector3f transformed(Tsg_map * point.getVector4fMap());
        alpha_s_list[i] = atan2f(-transformed(2), transformed(1));

    }

    int index;
    float alpha;
    int votes;

    map->searchBestMatch(hash_list, alpha_s_list, index, alpha, votes);

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
