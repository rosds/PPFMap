#ifndef PPFMAP_UTILS_HH__
#define PPFMAP_UTILS_HH__

#include <cuda_runtime.h>
#include <pcl/cuda/pcl_cuda_base.h>

#include <PPFMap/murmur.h>

#define PI_32F static_cast<float>(M_PI)
#define TWO_PI_32F static_cast<float>(2.0 * M_PI)
#define FOUR_PI_32F static_cast<float>(4.0 * M_PI)

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

    /** \brief Returns the angle between two vectors.
     *  \param[in] a First vector.
     *  \param[in] b Second vector.
     *  \return The angle between the two vectors. The value of the angle is 
     *  allways between 0 and pi.
     */
    __device__ __host__
    inline float angleBetween(const float3& a, const float3& b) {
        // Normalize input vectors
        const float3 a_unit = normalize(a);
        const float3 b_unit = normalize(b);
        const float3 c = cross(a_unit, b_unit);
        return atan2f(length(c), dot(a_unit, b_unit));
    }

    /** \brief Compute the Point Pair Feature between two points.
     *  \param[in] p1 First 3D point.
     *  \param[in] n1 The normal of the first point.
     *  \param[in] p2 Second 3D point.
     *  \param[in] n2 The normal of the second point.
     *  \param[out] f1 The first component of the PPF feature vector.
     *  \param[out] f2 The second component of the PPF feature vector.
     *  \param[out] f3 The third component of the PPF feature vector.
     *  \param[out] f4 The fourth component of the PPF feature vector.
     */
    __device__ __host__
    inline void computePPFFeature(const float3& p1, const float3& n1,
                                  const float3& p2, const float3& n2,
                                  float& f1, float& f2, float& f3, float& f4) {
    
        float3 d = make_float3(p2.x - p1.x,
                               p2.y - p1.y,
                               p2.z - p1.z);

        const float norm = length(d);

        if (norm != 0.0f) {
            d.x /= norm;
            d.y /= norm;
            d.z /= norm;
        } else {
            d = make_float3(0.0f, 0.0f, 0.0f);
        }

        // These 4 components should always be positive
        f1 = norm;
        f2 = ppfmap::angleBetween(d, n1);
        f3 = ppfmap::angleBetween(d, n2);
        f4 = ppfmap::angleBetween(n1, n2);
    }

    template <typename PointT, typename NormalT>
    __device__ __host__
    inline uint32_t computePPFFeatureHash(const PointT& r, const NormalT& r_n,
                                          const PointT& p, const NormalT& p_n,
                                          const float disc_dist,
                                          const float disc_angle) {
        return computePPFFeatureHash<float3, float3>(
                pointToFloat3(r), normalToFloat3(r_n),
                pointToFloat3(p), normalToFloat3(p_n),
                disc_dist, disc_angle);
    }


    template <>
    __device__ __host__
    inline uint32_t computePPFFeatureHash<float3, float3> (
        const float3& p1, const float3& n1,
        const float3& p2, const float3& n2,
        const float disc_dist, const float disc_angle) {

        // Compute the vector between the points
        float3 d = make_float3(p2.x - p1.x, p2.y - p1.y, p2.z - p1.z);

        // Compute the 4 components of the ppf feature
        const float f1 = length(d);
        const float f2 = ppfmap::angleBetween(d, n1);
        const float f3 = ppfmap::angleBetween(d, n2);
        const float f4 = ppfmap::angleBetween(n1, n2);

        // Discretize the PPF Feature before hashing
        uint32_t feature[4];
        feature[0] = static_cast<uint32_t>(f1 / disc_dist);
        feature[1] = static_cast<uint32_t>(f2 / disc_angle);
        feature[2] = static_cast<uint32_t>(f3 / disc_angle);
        feature[3] = static_cast<uint32_t>(f4 / disc_angle);

        // Return the hash of the feature.
        return murmurppf(feature);
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
    inline void getAlignmentToX(const PointT& point,
                                const NormalT& normal,
                                float (*affine)[12]) {
        getAlignmentToX<float3, float3>(pointToFloat3(point), normalToFloat3(normal), affine);
    }
    
    
    template<>
    __host__ __device__
    inline void getAlignmentToX<float3, float3> (
        const float3& point, const float3& normal, float (*affine)[12]) {
    
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

        // Calculate the angle between the normal and the X axis.
        float angle = angleBetween(normal, make_float3(1.0f, 0.0f,0.0f));

        // Normalize vector
        float norm = sqrt(v * v + w * w);
        v /= norm;
        w /= norm;

        // First row of rotation matrix
        (*affine)[0] = cosf(angle); 
        (*affine)[1] = - w * sinf(angle); 
        (*affine)[2] = v * sinf(angle); 

        // Second row of rotation matrix
        (*affine)[4] = w * sinf(angle);
        (*affine)[5] = v * v + w * w * cosf(angle); 
        (*affine)[6] = v * w * (1.0f - cosf(angle)); 

        // Third row of rotation matrix
        (*affine)[8] = - v * sinf(angle);
        (*affine)[9] = v * w * (1.0f - cosf(angle)); 
        (*affine)[10] = w * w + v * v * cosf(angle); 

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

    __device__ __host__
    inline float computeAlpha(const float3& p, const float T[12]) {
            // Compute the alpha angle
            float d_y = p.x * T[4] + p.y * T[5] + p.z * T[6] + T[7];
            float d_z = p.x * T[8] + p.y * T[9] + p.z * T[10] + T[11];
            return atan2f(-d_z, d_y);
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
            return length(point - ref_point);
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
