#include <PPFMap/Map.h>
#include <PPFMap/PPFEstimationKernel.h>

__constant__ float alignment_transformation[12];


ppfmap::Map::Map(const pcl::cuda::PointCloudSOA<pcl::cuda::Host>::Ptr cloud,
                 const pcl::cuda::PointCloudSOA<pcl::cuda::Host>::Ptr normals,
                 const float disc_dist,
                 const float disc_angle)
    : discretization_distance(disc_dist)
    , discretization_angle(disc_angle) {

    const size_t number_of_points = cloud->size();

    pcl::cuda::PointCloudSOA<pcl::cuda::Device> d_cloud;
    pcl::cuda::PointCloudSOA<pcl::cuda::Device> d_normals;

    d_cloud << *cloud;
    d_normals << *normals;

    ppf_codes.resize(number_of_points * number_of_points);

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

        // Set the transformation to the constant memory of the gpu.
        cudaMemcpyToSymbol(alignment_transformation, affine, 12 * sizeof(float));

        ppfmap::PPFEstimationKernel<pcl::cuda::Device> ppfe(point_position,
                                                            point_normal,
                                                            i,
                                                            discretization_distance,
                                                            discretization_angle,
                                                            alignment_transformation);

        thrust::transform(d_cloud.zip_begin(), d_cloud.zip_end(),
                          d_normals.zip_begin(),
                          ppf_codes.begin() + i * cloud->size(),
                          ppfe);

    }

    for (int i = 0; i < ppf_codes.size(); i++) {
        const uint64_t code = ppf_codes[i];

        uint32_t hk = static_cast<uint32_t>(code >> 32);
        uint32_t id = static_cast<uint32_t>(code >> 16 & 0xFFFF);
        uint32_t angle = static_cast<uint32_t>(code & 0xFFFF);

        std::cout << hk << " | " << id << " | " << angle << std::endl;
    }
}
