#ifndef PPFMAP_MAP_HH__
#define PPFMAP_MAP_HH__

#include <iostream>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/binary_search.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <boost/shared_ptr.hpp>

#include <PPFMap/utils.h>


namespace ppfmap {


/** \brief Contains the search structures for PPF Features of a particular 
 * model cloud.
 *
 *  This class computes the PPF features from a specified model cloud and 
 *  builds the corresponding search structures. These search structures remain 
 *  in the memory from the CUDA Device. It is important to keep in mind that 
 *  these structures grow quadratically with respect to the number of points in 
 *  the model cloud. In other words, for an N size cloud, there are NxN PPF 
 *  features to compute and save.
 */
class Map {
public:
    typedef boost::shared_ptr<Map> Ptr;

    /** \brief Empty constructor.
     */
    Map() 
        : discretization_distance(0.0f)
        , discretization_angle(0.0f)
        , max_votes(0) 
        , cloud_diameter(0.0f) {}

    /** \brief Computes the PPF features for the input cloud.
     *  \param[in] h_points Host vector with the 3D information of the points.
     *  \param[in] h_normals Host vector with the normals of each point.
     *  \param[in] disc_dist Discretization factor for pair distance.
     *  \param[in] disc_angle Discretization factor for angles.
     */
    Map(const thrust::host_vector<float3>& h_points,
        const thrust::host_vector<float3>& h_normals,
        const float disc_dist,
        const float disc_angle);

    virtual ~Map() {}

    /** \brief Performs the voting and accumulation for the ppf list provided 
     * and returns the best point index and resulting alpha.
     *  \param[in] hash_list List of hashed ppf features to query
     *  \param[in] alpha_s Angle to align the reference point to the x axis.
     *  \param[out] m_idx Best matching index in Hough voting space.
     *  \param[out] alpha Resulting angle after combining the alpha_s and 
     *  alpha_m.
     *  \param[out] max_votes The number of pairs supporting the m_idx and alpha 
     *  parameters.
     */
    void searchBestMatch(const thrust::host_vector<uint32_t> hash_list, 
                         const thrust::host_vector<float> alpha_s_list,
                         int& m_idx, float& alpha, int& max_votes);

    /** \brief Get lastest pair distance possible in the cloud.
     *  \return The largest distance between pairs in the cloud.
     */
    float getCloudDiameter() { return cloud_diameter; }

    /** \brief Returns the number of features stored in the map.
     *  \return The number of feature stored in the map.
     */
    std::size_t size() { return ppf_codes.size(); }

private:
    
    const float discretization_distance;
    const float discretization_angle;

    std::size_t max_votes;
    float cloud_diameter;

    thrust::device_vector<uint64_t> ppf_codes;
    thrust::device_vector<uint32_t> hash_keys;
    thrust::device_vector<uint32_t> ppf_index;
    thrust::device_vector<uint32_t> ppf_count;
}; // class Map

} // namespace ppfmap

#endif // PPFMAP_MAP_HH__
