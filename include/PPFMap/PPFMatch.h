#ifndef PPFMAP_PPFMATCH_HH__
#define PPFMAP_PPFMATCH_HH__

#include <pcl/common/common_headers.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/correspondence.h>

#include <thrust/host_vector.h>

#include <PPFMap/utils.h>
#include <PPFMap/Map.h>
#include <PPFMap/ppf_cuda_calls.h>


namespace ppfmap {


/** \brief Represents a pose supported by a correspondence.
 */
struct Pose {
    int votes;
    Eigen::Affine3f t;
    pcl::Correspondence c;
};


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
     *  \param[in] disc_dist Discretization distance for the point pairs.
     *  \param[in] disc_angle Discretization angle for the ppf features.
     */
    PPFMatch(const float disc_dist, const float disc_angle)
        : discretization_distance(disc_dist)
        , discretization_angle(disc_angle)
        , translation_threshold(0.1f)
        , rotation_threshold(12.0f / 180.0f * static_cast<float>(M_PI))
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

    /** \brief Search of the model in an scene cloud and returns the 
     * correspondences and the transformation to the scene.
     *
     *  \param[in] cloud Point cloud of the scene.
     *  \param[in] normals Normals of the scene cloud.
     *  \param[out] trans Affine transformation from to model to the scene.
     *  \param[out] correspondence Supporting correspondences from the scene to 
     *  the model.
     */
    void detect(const PointCloudPtr cloud, const NormalsPtr normals, 
                Eigen::Affine3f& trans, 
                pcl::Correspondences& correspondences);

private:

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
    int getPose(const int reference_index,
                const std::vector<int>& indices,
                const PointCloudPtr cloud,
                const NormalsPtr cloud_normals,
                const float affine_s[12],
                Pose* final_pose);

    /** \brief True if poses are similar given the translation and rotation 
     * thresholds.
     *  \param[in] t1 First pose.
     *  \param[in] t2 Second pose.
     *  \return True if the transformations are similar
     */
    bool similarPoses(const Eigen::Affine3f &t1, const Eigen::Affine3f& t2);

    void clusterPoses(const std::vector<Pose>& poses, Eigen::Affine3f& trans, pcl::Correspondences& corr);

    bool model_map_initialized;
    const float discretization_distance;
    const float discretization_angle;
    const float translation_threshold;
    const float rotation_threshold;

    PointCloudPtr model_;
    NormalsPtr normals_;
    ppfmap::Map::Ptr model_ppf_map;
};

} // namespace ppfmap

#include <PPFMap/impl/PPFMatch.hpp>

#endif // PPFMAP_PPFMATCH_HH__
