#ifndef PPFMAP_PPF_ESTIMATION_KERNEL_HH__
#define PPFMAP_PPF_ESTIMATION_KERNEL_HH__

namespace ppfmap {

    template <template <typename> class Storage>
    struct PPFEstimationKernel {
    
        const float3 point_position;
        const float3 point_normal;
        const int point_index;
        const float discretization_distance;
        const float discretization_angle;
        const float* alignment_transformation;

        PPFEstimationKernel(const float3 position,
                            const float3 normal,
                            const int index,
                            const float disc_dist,
                            const float disc_angle,
                            const float* transformation)
            : point_position(position)
            , point_normal(normal)
            , point_index(index)
            , discretization_distance(disc_dist)
            , discretization_angle(disc_angle)
            , alignment_transformation(transformation) {}

        template <typename T>
        __host__ __device__
        uint64_t operator()(const T position, const T normal) const {

            // Distance vector between points position
            float d_x = thrust::get<0>(position) - point_position.x;
            float d_y = thrust::get<1>(position) - point_position.y;
            float d_z = thrust::get<2>(position) - point_position.z;

            // First feature: The distance between the points.
            float f1 = sqrt(d_x * d_x + d_y * d_y + d_z * d_z);

            // Normalize direction vector.
            d_x /= f1;
            d_y /= f1;
            d_z /= f1;

            //  Second feature: the angle between the reference normal and 
            //  the direction vector.
            float f2 = acosf(
                d_x * point_normal.x + 
                d_y * point_normal.y + 
                d_z * point_normal.z
            );
             
            //  Third feature: the angle between the point normal and 
            //  the direction vector.
            float f3 = acosf(
                d_x * thrust::get<0>(normal) + 
                d_y * thrust::get<1>(normal) + 
                d_z * thrust::get<2>(normal)
            );
             
            //  Fourth feature: the angle between the two normals.
            float f4 = acosf(
                point_normal.x * thrust::get<0>(normal) + 
                point_normal.y * thrust::get<1>(normal) + 
                point_normal.z * thrust::get<2>(normal)
            );

            // Discretize the feature
            uint32_t d1 = static_cast<uint32_t>(f1 / discretization_distance);
            uint32_t d2 = static_cast<uint32_t>(f2 / discretization_angle);
            uint32_t d3 = static_cast<uint32_t>(f3 / discretization_angle);
            uint32_t d4 = static_cast<uint32_t>(f4 / discretization_angle);

            // Compute the hash key
            uint32_t hk = d1 ^ d2 ^ d3 ^ d4;

            d_x = thrust::get<0>(position) * alignment_transformation[0] + 
                  thrust::get<1>(position) * alignment_transformation[1] + 
                  thrust::get<2>(position) * alignment_transformation[2] + 
                  alignment_transformation[3];
            d_y = thrust::get<0>(position) * alignment_transformation[4] + 
                  thrust::get<1>(position) * alignment_transformation[5] + 
                  thrust::get<2>(position) * alignment_transformation[6] + 
                  alignment_transformation[7];
            d_z = thrust::get<0>(position) * alignment_transformation[8] + 
                  thrust::get<1>(position) * alignment_transformation[9] + 
                  thrust::get<2>(position) * alignment_transformation[10] + 
                  alignment_transformation[11];

            uint16_t id = static_cast<uint16_t>(point_index);
            uint16_t alpha = static_cast<int16_t>(-atan2f(-d_z, d_y) / discretization_angle);

            return (static_cast<uint64_t>(hk) << 32) | 
                   (static_cast<uint64_t>(id) << 16) | 
                   (static_cast<uint64_t>(alpha));
        }
    };

} // namespace ppfmap

#endif // PPFMAP_PPF_ESTIMATION_KERNEL_HH__
