#ifndef PPFMAP_STD_MAP_HH__
#define PPFMAP_STD_MAP_HH__

#include <tuple>
#include <unordered_map>

#include <pcl/common/common_headers.h>

namespace ppfmap {

class StdMap {
    public:
        typedef std::pair<int, float> PairInfo;
        typedef struct {
            union {
                int data[4];
                int f1;
                int f2;
                int f3;
                int f4;
            };
        } DiscretizedPPF;

        void trainModel(const pcl::PointCloud<pcl::PointNormal>::Ptr& cloud) {
        
        
        }

    private:
        std::unordered_multimap<DiscretizedPPF, PairInfo> map;
};

}; // namespace ppfmap

#endif // PPFMAP_STD_MAP_HH__
