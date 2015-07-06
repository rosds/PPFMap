#ifndef PPFMAP_PPF_ESTIMATION_KERNEL_HH__
#define PPFMAP_PPF_ESTIMATION_KERNEL_HH__

#include <PPFMap/utils.h>

__constant__ float affine[12];


namespace ppfmap {

    /** \brief Functor for calcukating the PPF Features and also the alpha 
     * angle.
     *
     *  This functor takes a particular reference point in the cloud and with 
     *  the operator(), it computes the PPF discretized feture and also the 
     *  alpha angle to a particular point.
     *
     *  Using a thrust::transform function, this functor is used to compute all 
     *  the ppf features in the cloud in parallel.
     */
    struct PPFEstimationKernel : public thrust::binary_function<float3, float3, uint64_t> {
    
        const float3 ref_point;
        const float3 ref_normal;
        const int point_index;
        const float discretization_distance;
        const float discretization_angle;

        /** \brief Constructor.
         *  \param[in] position Reference point position.
         *  \param[in] normal Reference point normal.
         *  \param[in] index Point index in the cloud.
         *  \param[in] disc_dist Discretization distance.
         *  \param[in] disc_angle Discretization angle.
         *  \param[in] transformation Pointer to the affine transformation that 
         *  aligns the pointes with respect to the reference point's normal.
         */
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

        __device__
        uint64_t operator()(const float3 point, const float3 point_normal) const {

            // Compute the hash key
            uint32_t hk = computePPFFeatureHash(ref_point, ref_normal,
                                                point, point_normal,
                                                discretization_distance,
                                                discretization_angle);

            uint16_t id = static_cast<uint16_t>(point_index);

            // Compute the alpha angle
            float d_y = point.x * affine[4] + 
                        point.y * affine[5] + 
                        point.z * affine[6] + affine[7];
            float d_z = point.x * affine[8] + 
                        point.y * affine[9] + 
                        point.z * affine[10] + affine[11];
            float alpha = atan2f(-d_z, d_y);

            // Discretize alpha
            alpha += static_cast<float>(M_PI); // alpha \in [0, 2pi]
            uint16_t alpha_disc = static_cast<uint16_t>(alpha / discretization_angle);
            
            return (static_cast<uint64_t>(hk) << 32) | 
                   (static_cast<uint64_t>(id) << 16) | 
                   (static_cast<uint64_t>(alpha_disc));
        }
    };

} // namespace ppfmap

#endif // PPFMAP_PPF_ESTIMATION_KERNEL_HH__
