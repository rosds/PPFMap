#include <PPFMap/Map.h>
#include <PPFMap/PPFEstimationKernel.h>


struct extract_hash_key : thrust::unary_function<uint64_t, uint32_t> {
    __host__ __device__
    uint32_t operator()(const uint64_t ppf_code) const {
        return static_cast<uint32_t>(ppf_code >> 32);
    }
};


ppfmap::Map::Map(const pcl::cuda::PointCloudSOA<pcl::cuda::Host>::Ptr cloud,
                 const pcl::cuda::PointCloudSOA<pcl::cuda::Host>::Ptr normals,
                 const float disc_dist,
                 const float disc_angle)
    : discretization_distance(disc_dist)
    , discretization_angle(disc_angle) {

    const size_t number_of_points = cloud->size();
    const size_t number_of_pairs = number_of_points * number_of_points;

    pcl::cuda::PointCloudSOA<pcl::cuda::Device> d_cloud;
    pcl::cuda::PointCloudSOA<pcl::cuda::Device> d_normals;

    d_cloud << *cloud;
    d_normals << *normals;

    ppf_codes.resize(number_of_pairs);

    for (int i = 0; i < number_of_points; i++) {
    
        const float3 point_position = make_float3(cloud->points_x[i],
                                                  cloud->points_y[i],
                                                  cloud->points_z[i]);

        const float3 point_normal = make_float3(normals->points_x[i],
                                                normals->points_y[i],
                                                normals->points_z[i]);

        // Calculate the angle between the normal and the X axis.
        float rotation_angle = acosf(point_normal.x);

        // Rotation axis lays on the plane y-z (i.e. u = 0)
        float v;
        float w;

        // The rotation axis is the cross product of the normal and the X axis. 
        if (point_normal.y == 0.0f && point_normal.z == 0.0f) {
            // Degenerate case, set the Y axis as the rotation axis
            v = 1.0f;
            w = 0.0f;
        } else {
            // This would be the cross product of the normal and the x axis.
            v = point_normal.z;
            w = - point_normal.y;
        }

        // Normalize vector
        float norm = sqrt(v * v + w * w);
        v /= norm;
        w /= norm;

        float affine[12];

        // First row of rotation matrix
        affine[0] = (v * v + w * w) * cosf(rotation_angle); 
        affine[1] = - w * sinf(rotation_angle); 
        affine[2] = v * sinf(rotation_angle); 

        // Second row of rotation matrix
        affine[4] = w * sinf(rotation_angle);
        affine[5] = v * v + w * w * cosf(rotation_angle); 
        affine[6] = v * w * (1.0f - cosf(rotation_angle)); 

        // Third row of rotation matrix
        affine[8] = - v * sinf(rotation_angle);
        affine[9] = v * w * (1.0f - cosf(rotation_angle)); 
        affine[10] = w * w + v * v * cosf(rotation_angle); 

        // Translation column
        affine[3] = - point_position.x * affine[0] 
                    - point_position.y * affine[1] 
                    - point_position.z * affine[2];

        affine[7] = - point_position.x * affine[4] 
                    - point_position.y * affine[5] 
                    - point_position.z * affine[6];

        affine[11] = - point_position.x * affine[8] 
                     - point_position.y * affine[9] 
                     - point_position.z * affine[10];

        ppfmap::PPFEstimationKernel<pcl::cuda::Device> 
            ppfe(point_position, point_normal, i,
                 discretization_distance,
                 discretization_angle,
                 affine);

        thrust::transform(d_cloud.zip_begin(), d_cloud.zip_end(),
                          d_normals.zip_begin(),
                          ppf_codes.begin() + i * cloud->size(),
                          ppfe);
    }

    thrust::sort(ppf_codes.begin(), ppf_codes.end());

    thrust::device_vector<uint32_t> hash_tmp(number_of_pairs);

    // copy the hash keys to a separate vector
    thrust::transform(
        ppf_codes.begin(), 
        ppf_codes.end(), 
        hash_tmp.begin(),
        extract_hash_key()
    );

    hash_keys.resize(number_of_pairs);
    ppf_count.resize(number_of_pairs);

    thrust::pair<thrust::device_vector<uint32_t>::iterator, 
                 thrust::device_vector<uint32_t>::iterator> end;

    // Count the number of similar keys
    end = thrust::reduce_by_key(hash_tmp.begin(), hash_tmp.end(),
                                thrust::make_constant_iterator(1),
                                hash_keys.begin(),
                                ppf_count.begin());

    const size_t unique_hash_keys = end.first - hash_keys.begin();

    // Fix the vectors to size
    hash_keys.resize(unique_hash_keys);
    hash_keys.shrink_to_fit();
    ppf_count.resize(unique_hash_keys);
    ppf_count.shrink_to_fit();

    ppf_index.resize(unique_hash_keys);

    // Set the array with the indices to the first instance of each key in the 
    // codes array.
    thrust::exclusive_scan(ppf_count.begin(), ppf_count.end(),
                           ppf_index.begin());

    max_votes = thrust::reduce(ppf_count.begin(), ppf_count.end(), 
                               0, thrust::maximum<uint32_t>());
}
