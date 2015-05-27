#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <pcl/common/common_headers.h>
#include <pcl/visualization/pcl_visualizer.h>


int main(int argc, char *argv[]) {

    pcl::PointCloud<pcl::PointXYZ>::Ptr model(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr scene(new pcl::PointCloud<pcl::PointXYZ>());
    
    pcl::io::loadPCDFile("../clouds/milk.pcd", *model);
    pcl::io::loadPCDFile("../clouds/milk_cartoon_all_small_clorox.pcd", *scene);

    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer());

    Eigen::Matrix4f trans = Eigen::Matrix4f::Identity();
    trans(0,3) = -2.0f;

    pcl::transformPointCloud(*model, *model, trans);

    viewer->addPointCloud(model, "model_cloud");
    viewer->addPointCloud(scene, "scene_cloud");

    while (!viewer->wasStopped()) {
        viewer->spinOnce();
    }

    return 0;
}
