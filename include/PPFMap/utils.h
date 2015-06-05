#ifndef PPFMAP_UTILS_HH__
#define PPFMAP_UTILS_HH__

#include <cuda_runtime.h>


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

    __device__ __host__
    inline uint32_t hashPPF(uint32_t dist, 
                            uint32_t a1, 
                            uint32_t a2, 
                            uint32_t a3) {
        return dist << 24 | a1 << 16 | a2 << 8 | a3;
    }

    __device__ __host__
    inline void computePPFFeature(const float3& r,
                                  const float3& r_n,
                                  const float3& p,
                                  const float3& p_n,
                                  float &f1, 
                                  float &f2, 
                                  float &f3, 
                                  float &f4) {

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
    }

    __device__ __host__
    inline uint32_t computePPFFeatureHash(const float3& r,
                                          const float3& r_n,
                                          const float3& p,
                                          const float3& p_n,
                                          const float disc_dist,
                                          const float disc_angle) {
        float f1, f2, f3, f4;
        computePPFFeature(r, r_n, p, p_n, f1, f2, f3, f4);
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
    inline void getAlignmentToX(const float3 point,
                                const float3 normal,
                                float (*affine)[12]) {
    
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

    }

} // namespace ppfmap

#endif // PPFMAP_UTILS_HH__
