#ifndef PPFMAP_DISCRETIZED_PPF_HH__
#define PPFMAP_DISCRETIZED_PPF_HH__

#include <PPFMap/murmur.h>
#include <PPFMap/utils.h>

namespace ppfmap {

    struct DiscretizedPPF {
        uint32_t data[4];    

        bool operator==(const DiscretizedPPF& ppf) const {
            return (ppf[0] == data[0]) && 
                   (ppf[1] == data[1]) && 
                   (ppf[2] == data[2]) && 
                   (ppf[3] == data[3]);
        }

        uint32_t operator[](const int i) const { return data[i]; }
        uint32_t& operator[](const int i) { return data[i]; }
    }; // struct DiscretizedPPF


    inline float angleBetween(const Eigen::Vector3f& a,
                              const Eigen::Vector3f& b) {
        const auto a_unit = a.normalized();
        const auto b_unit = b.normalized();
        const auto c = a.cross(b);
        return atan2f(c.norm(), a.dot(b));
    }


    template <typename PointT, typename NormalT>
    DiscretizedPPF computePPFDiscretized(const PointT& p1, const NormalT& pn1, 
                                         const PointT& p2, const NormalT& pn2,
                                         const float distance_step,
                                         const float angle_step) {

        const Eigen::Vector3f d = p2.getVector3fMap() - p1.getVector3fMap();
        const Eigen::Vector3f n1 = pn1.getNormalVector3fMap();
        const Eigen::Vector3f n2 = pn2.getNormalVector3fMap();

        // Compute the four PPF components
        const float f1 = d.norm();
        const float f2 = ppfmap::angleBetween(d, n1);
        const float f3 = ppfmap::angleBetween(d, n2);
        const float f4 = ppfmap::angleBetween(n1, n2);

        // Discretize the feature and return it
        DiscretizedPPF ppf;
        ppf[0] = static_cast<uint32_t>(f1 / distance_step);
        ppf[1] = static_cast<uint32_t>(f2 / angle_step);
        ppf[2] = static_cast<uint32_t>(f3 / angle_step);
        ppf[3] = static_cast<uint32_t>(f4 / angle_step);
        return ppf;
    }

} // namespace ppfmap

namespace std {
    template <>
    struct hash<ppfmap::DiscretizedPPF> {
        std::size_t operator()(const ppfmap::DiscretizedPPF& ppf) const {
            return murmurppf(ppf.data); 
        }
    }; // struct hash
} // namespace std

#endif // PPFMAP_DISCRETIZED_PPF_HH__
