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

} // namespace ppfmap

#endif // PPFMAP_POSE_HH__
