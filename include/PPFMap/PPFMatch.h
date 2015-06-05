#ifndef PPFMAP_PPFMATCH_HH__
#define PPFMAP_PPFMATCH_HH__

#include <pcl/common/common_headers.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <thrust/host_vector.h>

#include <PPFMap/utils.h>
#include <PPFMap/Map.h>
#include <PPFMap/ppf_cuda_calls.h>


namespace ppfmap {

/** \brief Implements the PPF features matching between two point clouds.
 *
 *  \tparam PointT Point type of the clouds.
 *  \tparam NormalT Normal type of the clouds.
 */
template <typename PointT, typename NormalT>
class PPFMatch {
public:
    typedef typename pcl::PointCloud<PointT>::Ptr PointCloudPtr;
    typedef typename pcl::PointCloud<NormalT>::Ptr NormalsPtr;

    /** \brief Constructor for the 
     *  \param disc_dist Discretization distance for the point pairs.
     *  \param disc_angle Discretization angle for the ppf features.
     */
    PPFMatch(const float disc_dist, const float disc_angle)
        : discretization_distance(disc_dist)
        , discretization_angle(disc_angle)
        , model_map_initialized(false) {}

    /** \brief Default destructor **/
    virtual ~PPFMatch() {}

    /** \brief Construct the PPF search structures for the model cloud.
     *  
     *  The model cloud contains the information about the object that is going 
     *  to be detected in the scene cloud. The necessary information to build 
     *  the search structure are the points and normals from the object.
     *
     *  \param[in] model Point cloud containing the model object.
     *  \param[in] normals Cloud with the normals of the object.
     */
    void setModelCloud(const PointCloudPtr model, const NormalsPtr normals);

    /** \brief Perform the voting and accumulation of the PPF features in the 
     * model and returns the model index with the most votes.
     *
     *  \param[in] point_index Index of the point.
     *  \param[in] cloud The pointer to the cloud where the queried point is.
     *  \param[in] cloud_normals The pointer to the normals of the cloud.
     *  \param[in] neighborhood_radius The radius to consider for building 
     *  pairs around the reference point.
     *  \return The index of the model point with the higher number of votes.
     */
    int findBestMatch(const int point_index,
                      const PointCloudPtr cloud,
                      const NormalsPtr cloud_normals,
                      const float neighborhood_radius,
                      Eigen::Affine3f& pose);

private:

    bool model_map_initialized;
    const float discretization_distance;
    const float discretization_angle;

    PointCloudPtr model_;
    NormalsPtr normals_;

    ppfmap::Map::Ptr model_ppf_map;
};

} // namespace ppfmap

#include <PPFMap/impl/PPFMatch.hpp>

#endif // PPFMAP_PPFMATCH_HH__
