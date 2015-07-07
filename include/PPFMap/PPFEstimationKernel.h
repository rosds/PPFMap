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
        float* alpha_m_array;
        uint64_t* ppf_codes;

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
                            thrust::device_vector<float>& alpha_m,
                            thrust::device_vector<uint64_t>& codes)
            : ref_point(position)
            , ref_normal(normal)
            , point_index(index)
            , discretization_distance(disc_dist)
            , discretization_angle(disc_angle)
            , number_of_points(n_points)
            , point_array(thrust::raw_pointer_cast(points.data()))
            , normal_array(thrust::raw_pointer_cast(normals.data())) 
            , alpha_m_array(thrust::raw_pointer_cast(alpha_m.data()))
            , ppf_codes(thrust::raw_pointer_cast(codes.data())) {
            
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

            // Compute the alpha angle
            float d_y = point.x * affine[4] + 
                        point.y * affine[5] + 
                        point.z * affine[6] + affine[7];
            float d_z = point.x * affine[8] + 
                        point.y * affine[9] + 
                        point.z * affine[10] + affine[11];

            float alpha = atan2f(-d_z, d_y);

            // Store the angle alpha separately and reference in
            alpha_m_array[i] = alpha;

            uint64_t code = (static_cast<uint64_t>(hk) << 32) | 
                            (static_cast<uint64_t>(point_index) << 16) | 
                            (static_cast<uint64_t>(i));

            // Save the code
            ppf_codes[point_index * number_of_points + i] = code;
        }
    };

} // namespace ppfmap

#endif // PPFMAP_PPF_ESTIMATION_KERNEL_HH__
