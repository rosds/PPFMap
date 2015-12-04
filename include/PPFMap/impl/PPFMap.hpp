#include <PPFMap/PPFMap.h>

template <typename PointT, typename NormalT>
void ppfmatch::PPFMap<PointT, NormalT>::compute(CloudPtr cloud, 
                                                NormalsPtr normals) {
  // Pair all the points in the cloud
  for (int i = 0; i < cloud->size(); i++) {

    const auto& p1 = cloud->at(i);
    const auto& n1 = normals_->at(i);

    if (!pcl::isFinite(p1) || !pcl::isFinite(n1)) {
      continue;
    }

    const auto Tmg = getTg(p1, n1);

    for (int j = 0; j < cloud->size(); j++) {

      const auto& p2 = cloud->at(j);
      const auto& n2 = normals_->at(j);

      if (i == j || !pcl::isFinite(p2) || !isFinite(n2)) {
        continue;
      }

      const auto ppf = computePPFDiscretized(p1, n1, p2, n2, 
                                             _distance_step, 
                                             _angle_step);

      const auto pt = Tmg * p2.getVector3fMap();
      float alpha_m = atan2(-pt(2), pt(1));

      VotePair vp(i, alpha_m);

      // Create the tuple for the map
      _map.insert(std::pair<DiscretizedPPF, VotePair>(ppf, vp));
    }
  }
}
