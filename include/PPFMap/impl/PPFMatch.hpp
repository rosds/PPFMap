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

