#ifndef PPFMAP_PPFMATCH_HH__
#define PPFMAP_PPFMATCH_HH__

#include <PPFMap/Map.h>

namespace ppfmap {

template <typename PointT, typename NormalT>
class PPFMatch {
public:
    typedef typename pcl::PointCloud<PointT>::Ptr PointCloudPtr;
    typedef typename pcl::PointCloud<NormalT>::Ptr NormalsPtr;

    PPFMatch() {}

    virtual ~PPFMatch() {}

    void setModelPointCloud(const PointCloudPtr model) {
        model_ = model; 
    }

    void setModelNormals(const NormalsPtr normals) {
        normals_ = normals;
    }

private:
    PointCloudPtr model_;
    NormalsPtr normals_;
    ppfmap::Map::Ptr model_ppf_map;
};

} // namespace ppfmap

#endif // PPFMAP_PPFMATCH_HH__
