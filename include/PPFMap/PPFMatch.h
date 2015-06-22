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
 *  This class's behavior is ruled basically by 5 parameters. Two of these 
 *  parameters affect the PPF Map structure and its performance. These are the 
 *  discretization step parameters that you set with the function 
 *  PPFMatch::setDiscretizationParameters.
 *
 *  These are two parameter; the discretization distance and discretization 
 *  angle steps. These two basically are meant to "group" similar features 
 *  together. Setting small discretization distance and angle will result in a 
 *  larger number of groups of features. The more groups, the more precision in 
 *  the geometry that is being saved in the map but lest robustness to noise. 
 *  On the other hand, big discretization steps for distance and angle will 
 *  create less groups with similar features, gut for generalization and 
 *  robustness against noise but the geometry of the model is then lost.
 *
 *  The next parameter is the percentage of the maximum radius for the pairs 
 *  neighborhood. The model has a maximum diameter; a pair of points with the 
 *  largest distance between them. Searching for pairs in the scene separated 
 *  with a distance larger than this maximum diameter makes no sense when 
 *  looking for possible matchings on the model cloud. Therefor, this 
 *  parameter represents the percentage of this maximum diameter to use as a 
 *  neighborhood radius search. You set this parameter with the function 
 *  PPFMatch::setMaxRadiusPercent.
 *
 *  Finally, the final two parameters are used in the pose clustering step. 
 *  These parameters represent thresholds to group similar poses. There are the 
 *  translation threshold and rotation threshold. The translation threshold 
 *  sets a limit on the distance between the translation parts of two poses for 
 *  being considered as similar. In the same way, the rotation threshold, sets 
 *  the limit in the angle for rotation between the two poses. You can set 
 *  these two with the function PPFMatch::setPoseClusteringThresholds.
 *
 *  \tparam PointT Point type of the clouds.
 *  \tparam NormalT Normal type of the clouds.
 */
template <typename PointT, typename NormalT>
class PPFMatch {
public:
    typedef typename pcl::PointCloud<PointT>::Ptr PointCloudPtr;
    typedef typename pcl::PointCloud<NormalT>::Ptr NormalsPtr;
    typedef boost::shared_ptr<PPFMatch<PointT, NormalT> > Ptr;

    /** \brief Constructor for the 
     *  \param[in] disc_dist Discretization distance for the point pairs.
     *  \param[in] disc_angle Discretization angle for the ppf features.
     */
    PPFMatch(const float disc_dist = 0.01f, 
             const float disc_angle = 12.0f / 180.0f * static_cast<float>(M_PI))
        : discretization_distance(disc_dist)
        , discretization_angle(disc_angle)
        , translation_threshold(0.7f)
        , rotation_threshold(30.0f / 180.0f * static_cast<float>(M_PI))
        , neighborhood_percentage(0.5f)
        , ref_point_indices(new std::vector<int>())
        , model_map_initialized(false)
        , use_indices(false) {}

    /** \brief Default destructor **/
    virtual ~PPFMatch() {}

    /** \brief Sets the percentage of the models diameter to use as maximum 
     * radius while searching pairs in the scene.
     *  \param[in] percent Float between 0 and 1 to represent the percentage of 
     *  the maximum radius possible when searching for the model in the secene.
     */
    void setMaxRadiusPercent(const float percent) {
        neighborhood_percentage = percent;
    }

    /** \brief Sets the discretization parameter for the PPF Map creation.
     *  \param[in] dist_disc Discretization distance step.
     *  \param[in] angle_disc Discretization angle step.
     */
    void setDiscretizationParameters(const float dist_disc,
                                     const float angle_disc) {
        discretization_distance = dist_disc;
        discretization_angle = angle_disc;
    }

    /** \brief Sets the translation and rotation thresholds for the pose 
     * clustering step.
     *  \param[in] translation_thresh Translation threshold.
     *  \param[in] rotation_thresh Rotation threshold.
     */
    void setPoseClusteringThresholds(const float translation_thresh,
                                     const float rotation_thresh) {
        translation_threshold = translation_thresh;
        rotation_threshold = rotation_thresh;
    }

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

    /** \brief Specify a vector of indices of points in the cloud to use as 
     * reference points for the detection task.
     *  \param[in] ind Shared pointer to a vector of indices.
     */
    void setReferencePointIndices(const pcl::IndicesPtr ind) {
        ref_point_indices = ind;
        use_indices = true;
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
    bool detect(const PointCloudPtr cloud, const NormalsPtr normals, 
                Eigen::Affine3f& trans, 
                pcl::Correspondences& correspondences);

private:

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
    Pose getPose(const int reference_index,
                 const std::vector<int>& indices,
                 const PointCloudPtr cloud,
                 const NormalsPtr cloud_normals,
                 const float affine_s[12]);

    /** \brief True if poses are similar given the translation and rotation 
     * thresholds.
     *  \param[in] t1 First pose.
     *  \param[in] t2 Second pose.
     *  \return True if the transformations are similar
     */
    bool similarPoses(const Eigen::Affine3f &t1, const Eigen::Affine3f& t2);

    /** \brief Returns the average pose and the correspondences for the most 
     * consistent cluster of poses.
     *  \param[in] poses Vector with the poses.
     *  \param[out] trans Average affine transformation for the biggest 
     *  cluster.
     *  \param[out] corr Vector of correspondences supporting the cluster.
     *  \return True if a cluster was found, false otherwise.
     */
    bool clusterPoses(const std::vector<Pose>& poses, Eigen::Affine3f& trans, pcl::Correspondences& corr);

    bool model_map_initialized;
    bool use_indices;
    float discretization_distance;
    float discretization_angle;
    float translation_threshold;
    float rotation_threshold;
    float neighborhood_percentage;

    PointCloudPtr model_;
    NormalsPtr normals_;
    ppfmap::Map::Ptr model_ppf_map;
    pcl::IndicesPtr ref_point_indices;
};

} // namespace ppfmap

#include <PPFMap/impl/PPFMatch.hpp>

#endif // PPFMAP_PPFMATCH_HH__
