#ifndef PPFMAP_MAP_HH__
#define PPFMAP_MAP_HH__

#include <iostream>
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <pcl/cuda/pcl_cuda_base.h>


namespace ppfmap {

class Map {
public:
    typedef boost::shared_ptr<Map> Ptr;

    /** \brief Empty constructor.
     */
    Map() : discretization_distance(0.0f) , discretization_angle(0.0f) {}

    /** \brief Computes the PPF features for the input cloud.
     *  \param[in] cloud Pointer to the point cloud.
     *  \param[in] normals Pointer to the normals of the cloud.
     *  \param[in] disc_dist Discretization factor for pair distance.
     *  \param[in] disc_angle Discretization factor for angles.
     */
    Map(const pcl::cuda::PointCloudSOA<pcl::cuda::Host>::Ptr cloud,
        const pcl::cuda::PointCloudSOA<pcl::cuda::Host>::Ptr normals,
        const float disc_dist,
        const float disc_angle);

    virtual ~Map() {}

private:
    
    const float discretization_distance;
    const float discretization_angle;

    thrust::device_vector<uint64_t> ppf_codes;

}; // class Map

} // namespace ppfmap

#endif // PPFMAP_MAP_HH__
