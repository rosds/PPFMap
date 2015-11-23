#ifndef PPFMAP_POSE_HH__
#define PPFMAP_POSE_HH__

#include <Eigen/Geometry>
#include <pcl/correspondence.h>

namespace ppfmap {

/** \brief Represents a pose supported by a correspondence.
 */
struct Pose {
    int votes;
    Eigen::Affine3f t;
    pcl::Correspondence c;
};

bool similarPoses(
    const Eigen::Affine3f& pose1, 
    const Eigen::Affine3f& pose2,
    const float translation_threshold,
    const float rotation_threshold) {

    // Translation difference.
    float position_diff = (pose1.translation() - pose2.translation()).norm();
    
    // Rotation angle difference.
    Eigen::AngleAxisf rotation_diff_mat(pose1.rotation().inverse() * pose2.rotation());
    float rotation_diff = fabsf(rotation_diff_mat.angle());

    return position_diff < translation_threshold &&
           rotation_diff < rotation_threshold;
}

void clusterPoses(
    const std::vector<Pose>& poses, 
    const float translation_threshold,
    const float rotation_threshold,
    Eigen::Affine3f &trans, 
    pcl::Correspondences& corr, 
    int& votes) {

    int cluster_idx;
    std::vector<std::pair<int, int> > cluster_votes;
    std::vector<std::vector<Pose> > pose_clusters;

    for (const auto& pose : poses) {

        bool found_cluster = false;

        cluster_idx = 0;
        for (auto& cluster : pose_clusters) {
            if (similarPoses(pose.t, cluster.front().t, translation_threshold, rotation_threshold)) {
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

    votes = cluster_votes.back().first;
}

} // namespace ppfmap

#endif // PPFMAP_POSE_HH__
