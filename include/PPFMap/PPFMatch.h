#ifndef PPFMAP_PPFMATCH_HH__
#define PPFMAP_PPFMATCH_HH__

#include <pcl/common/common_headers.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <thrust/host_vector.h>

#include <PPFMap/utils.h>
#include <PPFMap/Map.h>
#include <PPFMap/ppf_cuda_calls.h>


namespace ppfmap {

template <typename PointT, typename NormalT>
class PPFMatch {
public:
    typedef typename pcl::PointCloud<PointT>::Ptr PointCloudPtr;
    typedef typename pcl::PointCloud<NormalT>::Ptr NormalsPtr;

    PPFMatch(const float disc_dist, const float disc_angle)
        : discretization_distance(disc_dist)
        , discretization_angle(disc_angle) {}

    virtual ~PPFMatch() {}

    void setModelPointCloud(const PointCloudPtr model) {
        model_ = model; 
    }

    void setModelNormals(const NormalsPtr normals) {
        normals_ = normals;
    }

    void initPPFSearchStruct();

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

    const float discretization_distance;
    const float discretization_angle;

    PointCloudPtr model_;
    NormalsPtr normals_;

    ppfmap::Map::Ptr model_ppf_map;
};

} // namespace ppfmap

#include <PPFMap/impl/PPFMatch.hpp>

#endif // PPFMAP_PPFMATCH_HH__
