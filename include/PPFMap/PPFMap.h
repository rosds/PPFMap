#ifndef PPFMAP_PPFMAP_HH__
#define PPFMAP_PPFMAP_HH__

#include <unordered_map>

#include <PPFMap/DiscretizedPPF.h>

namespace ppfmap {

/** \brief Point Pair Features Map for quick feature matching.
 *  \tparam PointT Point type.
 *  \tparam NormalT Normal type.
 */
template <typename PointT, typename NormalT>
class PPFMap {
  public:
    typedef std::shared_ptr<PPFMap<PointT, NormalT> > Ptr;

    typedef std::pair<int, float> VotePair;
    typedef typename pcl::PointCloud<PointT>::ConstPtr CloudPtr;
    typedef typename pcl::PointCloud<NormalT>::ConstPtr NormalsPtr;
    typedef std::unordered_multimap<DiscretizedPPF, VotePair> Map;

    /** \brief Empty constructor. **/
    PPFMap()
        : _distance_step(0.002f), _angle_step(12.0f / 180.0f * M_PI) {}

    /** \brief Constructor with distance and angle discretization **/
    PPFMap(float distance_step, float angle_step)
        : _distance_step(distance_step), _angle_step(angle_step) {}

    /** \brief Setter for the distance discretization parameter.
     *  \param[in] distance_set Distance discretization step.
     */
    inline void setDiscretizationDistance(float distance_step)
        { _distance_step = distance_step; }

    /** \brief Setter for the angle discretization parameter.
     *  \param[in] angle_step Angle discretization step.
     */
    inline void setDiscretizationAngle(float angle_step)
        { _angle_step = angle_step; }

    /** \brief Compute the map for the input model.
     *  \param[in] cloud Point cloud of the model.
     *  \param[in] normals Cloud with the model's normals.
     */
    void compute(CloudPtr cloud, NormalsPtr normals);

  private:
    float _distance_step;
    float _angle_step;
    Map _map;
}; // class PPFMap

} // namespace ppfmap

#include <PPFMap/impl/PPFMap.hpp>

#endif // PPFMAP_PPFMAP_HH__
