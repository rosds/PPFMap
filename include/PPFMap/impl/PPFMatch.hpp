#include <PPFMap/PPFMatch.h>

template <typename PointT, typename NormalT>
void 
ppfmap::PPFMatch<PointT, NormalT>::setModelCloud(const PointCloudPtr& model, 
                                                const NormalsPtr& normals) {

    model_ = model;
    normals_ = normals;
    float affine[12];
    
    // Pair all the points in the cloud
    for (int i = 0; i < model_->size(); i++) {

        const auto& p1 = model_->at(i);
        const auto& n1 = normals_->at(i);

        if (!pcl::isFinite(p1) || !pcl::isFinite(n1)) {
            continue;
        }

        ppfmap::getAlignmentToX(p1, n1, affine);
        Eigen::Map<Eigen::Matrix<float, 3, 4, Eigen::RowMajor> > Tmg(affine);

        for (int j = 0; j < model_->size(); j++) {
            
            const auto& p2 = model_->at(j);
            const auto& n2 = normals_->at(j);

            if (i == j || !pcl::isFinite(p2) || !isFinite(n2)) {
                continue;
            }

            const auto ppf = computePPFDiscretized(p1, n1, p2, n2, 
                                                   distance_step, 
                                                   angle_step);

            const auto pt = Tmg * p2.getVector3fMap();
            float alpha_m = atan2(-pt(2), pt(1));

            VotePair vp(i, alpha_m);

            // Create the tuple for the map
            map.insert(std::pair<DiscretizedPPF, VotePair>(ppf, vp));
        }
    }

    // Compute cloud diameter
    pcl::PointNormal min, max;
    pcl::getMinMax3D(*model, min, max);
    model_diameter = (max.getVector3fMap() - min.getVector3fMap()).norm();
}
