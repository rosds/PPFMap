#include <PPFMap/ppf.h>

template <typename PointT, typename NormalT>
ppfmap::PPFFeature::PPFFeature(
    const PointT& p1, const NormalT& n1,
    const PointT& p2, const NormalT& n2) {

    computePPFFeature(p1, n1, p2, n2);
}


template <typename PointT, typename NormalT>
void ppfmap::PPFFeature::computePPFFeature(
    const PointT& p1, const NormalT& n1,
    const PointT& p2, const NormalT& n2) {

    computePPFFeature<float3, float3>(
            pointToFloat3(p1), normalToFloat3(n1), 
            pointToFloat3(p2), normalToFloat3(n2));
}


template <>
void ppfmap::PPFFeature::computePPFFeature<float3, float3> (
    const float3& p1, const float3& n1,
    const float3& p2, const float3& n2) {

    float3 d = make_float3(p2.x - p1.x, p2.y - p1.y, p2.z - p1.z);
    const float norm = ppfmap::norm(d);

    // Normalize the vector between the two points
    if (norm != 0.0f) {
        d.x /= norm;
        d.y /= norm;
        d.z /= norm;
    } else {
        d = make_float3(0.0f, 0.0f, 0.0f);
    }

    f1 = norm;
    f2 = acos(ppfmap::dot(d, n1));
    f3 = acos(ppfmap::dot(d, n2));
    f4 = acos(ppfmap::dot(n1, n2));
}
