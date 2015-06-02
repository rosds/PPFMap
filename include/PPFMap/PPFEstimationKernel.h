#ifndef PPFMAP_PPF_ESTIMATION_KERNEL_HH__
#define PPFMAP_PPF_ESTIMATION_KERNEL_HH__

#include <PPFMap/utils.h>

__constant__ float affine[12];


namespace ppfmap {

    template <template <typename> class Storage>
    struct PPFEstimationKernel {
    
        const float3 ref_point;
        const float3 ref_normal;
        const int point_index;
        const float discretization_distance;
        const float discretization_angle;

        PPFEstimationKernel(const float3 position,
                            const float3 normal,
                            const int index,
                            const float disc_dist,
                            const float disc_angle,
                            float *transformation)
            : ref_point(position)
            , ref_normal(normal)
            , point_index(index)
            , discretization_distance(disc_dist)
            , discretization_angle(disc_angle) {
            
            // Set the transformation to the constant memory of the gpu.
            cudaMemcpyToSymbol(affine, transformation, 12 * sizeof(float));
        }


        template <typename T> __device__
        uint64_t operator()(const T position, const T normal) const {

            float3 point = make_float3(thrust::get<0>(position),
                                       thrust::get<1>(position),
                                       thrust::get<2>(position));

            float3 point_normal = make_float3(thrust::get<0>(normal),
                                              thrust::get<1>(normal),
                                              thrust::get<2>(normal));

            // Compute the hash key
            uint32_t hk = computePPFFeatureHash(ref_point, ref_normal,
                                                point, point_normal,
                                                discretization_distance,
                                                discretization_angle);

            float d_y = point.x * affine[4] + 
                        point.y * affine[5] + 
                        point.z * affine[6] + affine[7];
            float d_z = point.x * affine[8] + 
                        point.y * affine[9] + 
                        point.z * affine[10] + affine[11];

            uint16_t id = static_cast<uint16_t>(point_index);
            uint16_t alpha = static_cast<int16_t>(atan2f(-d_z, d_y) / discretization_angle);
            
            return (static_cast<uint64_t>(hk) << 32) | 
                   (static_cast<uint64_t>(id) << 16) | 
                   (static_cast<uint64_t>(alpha));
        }
    };

} // namespace ppfmap

#endif // PPFMAP_PPF_ESTIMATION_KERNEL_HH__
