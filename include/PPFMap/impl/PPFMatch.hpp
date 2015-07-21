#include <PPFMap/PPFMatch.h>

template <typename PointT, typename NormalT>
void 
ppfmap::PPFMatch<PointT, NormalT>::setModelCloud(const PointCloudPtr& model, 
                                                const NormalsPtr& normals) {

    model_ = model;
    normals_ = normals;
    float affine[12];
    
    // Pair all the points in the cloud
    for (int i = 0; i < model_->size(); i++) {

        const auto& p1 = model_->at(i);
        const auto& n1 = normals_->at(i);

        if (!pcl::isFinite(p1) || !pcl::isFinite(n1)) {
            continue;
        }

        const auto Tmg = getTg(p1, n1);

        for (int j = 0; j < model_->size(); j++) {
            
            const auto& p2 = model_->at(j);
            const auto& n2 = normals_->at(j);

            if (i == j || !pcl::isFinite(p2) || !isFinite(n2)) {
                continue;
            }

            const auto ppf = computePPFDiscretized(p1, n1, p2, n2, 
                                                   distance_step, 
                                                   angle_step);

            const auto pt = Tmg * p2.getVector3fMap();
            float alpha_m = atan2(-pt(2), pt(1));

            VotePair vp(i, alpha_m);

            // Create the tuple for the map
            map.insert(std::pair<DiscretizedPPF, VotePair>(ppf, vp));
        }
    }

    // Compute cloud diameter
    pcl::PointNormal min, max;
    pcl::getMinMax3D(*model, min, max);
    model_diameter = (max.getVector3fMap() - min.getVector3fMap()).norm();
}

template <typename PointT, typename NormalT>
bool 
ppfmap::PPFMatch<PointT, NormalT>::detect(const PointCloudPtr scene, 
                                          const NormalsPtr normals, 
                                          std::vector<Pose>& poses) {

    const int angle_bins = 30;
    std::vector<int> accumulator(model_->size() * angle_bins, 0);

    pcl::KdTreeFLANN<PointT> scene_search;
    scene_search.setInputCloud(scene);

    pcl::IndicesPtr ref_indices;
    if (use_indices) {
        ref_indices = ref_point_indices; 
    } else {
        ref_indices = pcl::IndicesPtr(new std::vector<int>(scene->size()));
        std::iota(ref_indices->begin(), ref_indices->end(), 0);
    }

    for (const auto& i : *ref_indices) {

        const auto& p1 = scene->at(i);
        const auto& n1 = normals->at(i);

        if (!pcl::isFinite(p1) || !pcl::isFinite(n1)) {
            continue;
        }

        const auto Tsg = getTg(p1, n1);

        // Loop through nearest neighbors
        std::vector<int> neighbor_indices;
        std::vector<float> distances;
        scene_search.radiusSearch(p1, neighborhood_percentage * model_diameter, neighbor_indices, distances);

        for (const auto& j : neighbor_indices) {

            const auto& p2 = scene->at(j);
            const auto& n2 = normals->at(j);

            if (i == j || !pcl::isFinite(p2) || !pcl::isFinite(n2)) {
                continue;
            }

            // Compute and discretize feature
            const auto ppf = computePPFDiscretized(p1, n1, p2, n2, 
                                                   distance_step, 
                                                   angle_step);

            // Compute the alpha_s angle
            const auto pt = Tsg * p2.getVector3fMap();
            float alpha_s = atan2(-pt(2), pt(1));

            auto similar_features = map.equal_range(ppf);

            // Accumulate the votes of similar features
            std::for_each(
                similar_features.first,
                similar_features.second,
                [&](const std::pair<DiscretizedPPF, VotePair>& match) {
                    const int model_i = match.second.model_i;
                    const float alpha_m = match.second.alpha_m;          

                    float alpha = alpha_m - alpha_s; 
                    const int alpha_bin = static_cast<int>(static_cast<float>(angle_bins) * ((alpha + 2.0f * static_cast<float>(M_PI)) / (4.0f * static_cast<float>(M_PI))));

                    // Count votes
                    accumulator[model_i * angle_bins + alpha_bin]++;
                }
            );
        }

        // Look for the winner
        int max_votes = 0;
        int max_votes_idx = 0;
        for (int k = 0; k < accumulator.size(); k++) {
            if (accumulator[k] > max_votes) {
                max_votes = accumulator[k];
                max_votes_idx = k;
            } 
            accumulator[k] = 0; // Set it to zero for next iteration
        } 

        int max_model_i = max_votes_idx / angle_bins;
        int max_alpha = max_votes_idx % angle_bins;

        const auto& model_point = model_->at(max_model_i);
        const auto& model_normal = normals_->at(max_model_i);
        const auto Tmg = getTg(model_point, model_normal);

        float angle = (static_cast<float>(max_alpha) / static_cast<float>(angle_bins)) * 4.0f * static_cast<float>(M_PI) - 2.0f * static_cast<float>(M_PI);

        Eigen::AngleAxisf rot(angle, Eigen::Vector3f::UnitX());
        
        // Compose the transformations for the final pose
        Eigen::Affine3f final_transformation(Tsg.inverse() * rot * Tmg);

        Pose pose;
        pose.t = final_transformation;
        pose.votes = max_votes;
        pose.c = pcl::Correspondence(i, max_model_i, 0.0f);

        poses.push_back(pose);
    }

    // Sort the pose vector by the poses votes
    std::sort(poses.begin(), poses.end(), 
        [](const Pose& a, const Pose& b) { return a.votes > b.votes; });
}


template <typename PointT, typename NormalT>
bool ppfmap::PPFMatch<PointT, NormalT>::detect(
    const PointCloudPtr scene, const NormalsPtr normals, 
    Eigen::Affine3f& trans, 
    pcl::Correspondences& correspondences,
    int& votes) {

    std::vector<Pose> poses;
    detect(scene, normals, poses);
    clusterPoses(
        poses, 
        translation_threshold, 
        rotation_threshold,
        trans, 
        correspondences, 
        votes);
}
