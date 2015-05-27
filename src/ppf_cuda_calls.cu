#include <PPFMap/ppf_cuda_calls.h>

ppfmap::Map::Ptr 
ppfmap::cuda::setPPFMap(const float *points, 
                        const float *normals,
                        const size_t n,
                        const float disc_dist,
                        const float disc_angle) {

    // Init PointCloudSOA structs
    pcl::cuda::PointCloudSOA<pcl::cuda::Host>::Ptr model(new pcl::cuda::PointCloudSOA<pcl::cuda::Host>);
    pcl::cuda::PointCloudSOA<pcl::cuda::Host>::Ptr model_normals(new pcl::cuda::PointCloudSOA<pcl::cuda::Host>);

    model->resize(n);
    model_normals->resize(n);

    for (int i = 0; i < n; i++) {
        model->points_x[i] = points[4 * i + 0]; 
        model->points_y[i] = points[4 * i + 1]; 
        model->points_z[i] = points[4 * i + 2]; 
    
        model_normals->points_x[i] = normals[4 * i + 0]; 
        model_normals->points_y[i] = normals[4 * i + 1]; 
        model_normals->points_z[i] = normals[4 * i + 2]; 
    }

    return boost::shared_ptr<Map>(new Map(model, model_normals, disc_dist, disc_angle));
}
