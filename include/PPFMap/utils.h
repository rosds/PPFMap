#ifndef PPFMAP_UTILS_HH__
#define PPFMAP_UTILS_HH__

#include <cuda_runtime.h>
#include <pcl/cuda/pcl_cuda_base.h>


namespace ppfmap {

    template <typename PointT>
    __device__ __host__
    float3 pointToFloat3(const PointT& p) {
        return make_float3(p.x, p.y, p.z); 
    }

    template <typename NormalT>
    __device__ __host__
    float3 normalToFloat3(const NormalT& p) {
        return make_float3(p.normal_x, p.normal_y, p.normal_z); 
    }

    __device__ __host__
    inline float dot(const float3& a, const float3& b) {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    __device__ __host__
    inline float norm(const float3& v) {
        return sqrt(ppfmap::dot(v, v));
    }

    /** \brief Concatenate each discretized component of the PPF vector into a 
     * 32 bit unsigned int.
     *  \param[in] f1 First PPF component.
     *  \param[in] f2 Second PPF component.
     *  \param[in] f3 Third PPF component.
     *  \param[in] f4 Fourth PPF component.
     *  \return Unsigned int with the concatenated components
     */
    __device__ __host__
    inline uint32_t hashPPF(uint32_t f1, uint32_t f2, uint32_t f3, uint32_t f4) {
        return f1 << 24 | f2 << 16 | f3 << 8 | f4;
    }


    template <typename PointT, typename NormalT>
    __device__ __host__
    inline uint32_t computePPFFeatureHash(const PointT& r, const NormalT& r_n,
                                          const PointT& p, const NormalT& p_n,
                                          const float disc_dist,
                                          const float disc_angle) {
        return computePPFFeatureHash<float3, float3>(pointToFloat3(r), normalToFloat3(r_n),
                                                     pointToFloat3(p), normalToFloat3(p_n),
                                                     disc_dist, disc_angle);
    }


    template <>
    __device__ __host__
    inline uint32_t computePPFFeatureHash<float3, float3> (
        const float3& r, const float3& r_n,
        const float3& p, const float3& p_n,
        const float disc_dist, const float disc_angle) {

        float f1, f2, f3, f4;
        float3 d = make_float3(p.x - r.x,
                               p.y - r.y,
                               p.z - r.z);

        const float norm = ppfmap::norm(d);

        d.x /= norm;
        d.y /= norm;
        d.z /= norm;

        f1 = norm;
        f2 = acos(ppfmap::dot(d, r_n));
        f3 = acos(ppfmap::dot(d, p_n));
        f4 = acos(ppfmap::dot(p_n, r_n));

        uint32_t d1 = static_cast<uint32_t>(f1 / disc_dist);
        uint32_t d2 = static_cast<uint32_t>(f2 / disc_angle);
        uint32_t d3 = static_cast<uint32_t>(f3 / disc_angle);
        uint32_t d4 = static_cast<uint32_t>(f4 / disc_angle);

        return hashPPF(d1, d2, d3, d4);
    }

    
    /** \brief Compute the affine transformation to align the point's normal to 
     * the X axis.
     *  \param[in] point The 3D position of the point.
     *  \param[in] normal The normal of the point.
     *  \param[out] affine Affine trasformation to align the 
     *  normal.
     */
    template <typename PointT, typename NormalT>
    __host__
    inline void getAlignmentToX(const PointT point,
                                const NormalT normal,
                                float (*affine)[16]) {
        getAlignmentToX<float3, float3>(pointToFloat3(point), normalToFloat3(normal), affine);
    }
    
    
    template<>
    __host__ __device__
    inline void getAlignmentToX<float3, float3> (
        const float3 point, const float3 normal, float (*affine)[16]) {
    
        // Calculate the angle between the normal and the X axis.
        float rotation_angle = acosf(normal.x);

        // Rotation axis lays on the plane y-z (i.e. u = 0)
        float v;
        float w;

        // The rotation axis is the cross product of the normal and the X axis. 
        if (normal.y == 0.0f && normal.z == 0.0f) {
            // Degenerate case, set the Y axis as the rotation axis
            v = 1.0f;
            w = 0.0f;
        } else {
            // This would be the cross product of the normal and the x axis.
            v = normal.z;
            w = - normal.y;
        }

        // Normalize vector
        float norm = sqrt(v * v + w * w);
        v /= norm;
        w /= norm;

        // First row of rotation matrix
        (*affine)[0] = (v * v + w * w) * cosf(rotation_angle); 
        (*affine)[1] = - w * sinf(rotation_angle); 
        (*affine)[2] = v * sinf(rotation_angle); 

        // Second row of rotation matrix
        (*affine)[4] = w * sinf(rotation_angle);
        (*affine)[5] = v * v + w * w * cosf(rotation_angle); 
        (*affine)[6] = v * w * (1.0f - cosf(rotation_angle)); 

        // Third row of rotation matrix
        (*affine)[8] = - v * sinf(rotation_angle);
        (*affine)[9] = v * w * (1.0f - cosf(rotation_angle)); 
        (*affine)[10] = w * w + v * v * cosf(rotation_angle); 

        // Translation column
        (*affine)[3] = - point.x * (*affine)[0] 
                       - point.y * (*affine)[1] 
                       - point.z * (*affine)[2];
                                        
        (*affine)[7] = - point.x * (*affine)[4] 
                       - point.y * (*affine)[5] 
                       - point.z * (*affine)[6];

        (*affine)[11] = - point.x * (*affine)[8] 
                        - point.y * (*affine)[9] 
                        - point.z * (*affine)[10];

        (*affine)[12] = 0.0f;
        (*affine)[13] = 0.0f;
        (*affine)[14] = 0.0f;
        (*affine)[15] = 1.0f;
    }


    /** \brief Functor structure used to compute the distance between two 
     * points.
     */
    struct compute_distance : public thrust::unary_function<float3, float> {
        const float3 ref_point;

        /** \brief Constructor.
         *  \param[in] point The point to use as reference for computing the 
         *  distance.
         */
        compute_distance(const float3 point) : ref_point(point) {}

        __host__ __device__ 
        float operator()(const float3 &point) const {
            return ppfmap::norm(point - ref_point);
        }
    };


    /** \brief Compute the farthest distance between a point and the rest of 
     * the point cloud.
     *  \param[in] point The point from which to look for the farthest 
     *  distance.
     *  \param[in] points PCL cuda storage containing the points as float3.
     */
    template <template <typename> class Storage>
    inline float maxDistanceToPoint(
        const float3 point, 
        const typename Storage<float3>::type& points) {
        return thrust::transform_reduce(points.begin(), points.end(),
                                        compute_distance(point), 0.0f,
                                        thrust::maximum<float>());
    
    }
} // namespace ppfmap

#endif // PPFMAP_UTILS_HH__

