#include <PPFMap/PPFMatch.h>

template <typename PointT, typename NormalT>
void ppfmap::PPFMatch<PointT, NormalT>::initPPFSearchStruct() {

    const std::size_t number_of_points = model_->size();

    std::unique_ptr<float[]> points(new float[3 * number_of_points]);
    std::unique_ptr<float[]> normals(new float[3 * number_of_points]);

    for (int i = 0; i < number_of_points; i++) {
        const auto& point = model_->at(i); 
        const auto& normal = normals_->at(i); 

        points[i * 3 + 0] = point.x;
        points[i * 3 + 1] = point.y;
        points[i * 3 + 2] = point.z;

        normals[i * 3 + 0] = normal.normal_x;
        normals[i * 3 + 1] = normal.normal_y;
        normals[i * 3 + 2] = normal.normal_z;
    }

    model_ppf_map = cuda::setPPFMap(points.get(), 
                                    normals.get(), 
                                    number_of_points,
                                    discretization_distance,
                                    discretization_angle);
}


template <typename PointT, typename NormalT>
int ppfmap::PPFMatch<PointT, NormalT>::findBestMatch(
    const int point_index,
    const PointCloudPtr cloud,
    const NormalsPtr cloud_normals,
    const float radius_neighborhood) {

    float affine[12];
    float f1, f2, f3, f4;
    uint32_t d1, d2, d3, d4;

    const auto& ref_point = cloud->at(point_index);
    const auto& ref_normal = cloud_normals->at(point_index);

    getAlignmentToX(ref_point, ref_normal, (float**)&affine);

    std::vector<int> indices;
    std::vector<float> distances;
    pcl::KdTreeFLANN<PointT> kdtree;
    kdtree.setInputCloud(cloud);
    kdtree.radiusSearch(ref_point, radius_neighborhood, indices, distances);

    thrust::host_vector<uint32_t> hash_list;
    thrust::host_vector<float> alpha_s_list;

    // Compute the PPF feature for all the pairs in the neighborhood
    for (const auto index : indices) {

        const auto& point = cloud->at(index);
        const auto& normal = cloud_normals->at(index);

        // Transform the point and compute the alpha_s
        float d_y = point.x * affine[4] + point.y * affine[5] + point.z * affine[6] + affine[7]; 
        float d_z = point.x * affine[8] + point.y * affine[9] + point.z * affine[10] + affine[11]; 
        float alpha_s = -atan2f(-d_z, d_y);
    
        computePPFFeature(ref_point, ref_normal,
                          point, normal,
                          f1, f2, f3, f4);

        d1 = static_cast<uint32_t>(f1 / discretization_distance);
        d2 = static_cast<uint32_t>(f2 / discretization_angle);
        d3 = static_cast<uint32_t>(f3 / discretization_angle);
        d4 = static_cast<uint32_t>(f4 / discretization_angle);

        uint32_t hash_key = d1 ^ d2 ^ d3 ^ d4;

        hash_list.push_back(hash_key);
        alpha_s_list.push_back(alpha_s);
    }

    int index;
    float alpha;
    model_ppf_map->searchBestMatch(hash_list, alpha_s_list, index, alpha);

    return index;
}
