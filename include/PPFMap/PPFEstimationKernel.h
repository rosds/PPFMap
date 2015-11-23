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
    struct PPFEstimationKernel {
    
        const int number_of_points;
        const float3 ref_point;
        const float3 ref_normal;
        const int point_index;
        const float discretization_distance;
        const float discretization_angle;
        const float3* point_array;
        const float3* normal_array;
        uint64_t* ppf_codes;
        const int angle_bins;

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
                            float *transformation,
                            const int n_points,
                            const thrust::device_vector<float3>& points,
                            const thrust::device_vector<float3>& normals,
                            thrust::device_vector<uint64_t>& codes)
            : ref_point(position)
            , ref_normal(normal)
            , point_index(index)
            , discretization_distance(disc_dist)
            , discretization_angle(disc_angle)
            , number_of_points(n_points)
            , point_array(thrust::raw_pointer_cast(points.data()))
            , normal_array(thrust::raw_pointer_cast(normals.data())) 
            , ppf_codes(thrust::raw_pointer_cast(codes.data()))
            , angle_bins(static_cast<int>(ceil(TWO_PI_32F / disc_angle))) {
            
            // Set the transformation to the constant memory of the gpu.
            cudaMemcpyToSymbol(affine, transformation, 12 * sizeof(float));
        }

        __device__
        void operator()(const int i) const {
            const float3& point = point_array[i];
            const float3& point_normal = normal_array[i];

            // Compute the hash key
            uint32_t hk = computePPFFeatureHash(ref_point, ref_normal,
                                                point, point_normal,
                                                discretization_distance,
                                                discretization_angle);

            const float alpha = computeAlpha(point, affine);

            uint16_t alpha_disc = static_cast<uint16_t>(angle_bins * (alpha + PI_32F) / TWO_PI_32F);

            uint64_t code = (static_cast<uint64_t>(hk) << 32) | 
                            (static_cast<uint64_t>(point_index & 0xFFFF) << 16) | 
                            (static_cast<uint64_t>(alpha_disc));

            // Save the code
            ppf_codes[point_index * number_of_points + i] = code;
        }
    };

} // namespace ppfmap

#endif // PPFMAP_PPF_ESTIMATION_KERNEL_HH__
