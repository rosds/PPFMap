#include <PPFMap/ppf.h>

ppfmap::PPFFeature::PPFFeature(const float3& p1, const float3& n1,
                               const float3& p2, const float3& n2) {

        float3 d = make_float3(p2.x - p1.x, p2.y - p1.y, p2.z - p1.z);
        const float norm = ppfmap::norm(d);

        // Normalize the vector between the two points
        d.x /= norm;
        d.y /= norm;
        d.z /= norm;

        f1 = norm;
        f2 = acos(ppfmap::dot(d, n1));
        f3 = acos(ppfmap::dot(d, n2));
        f4 = acos(ppfmap::dot(n1, n2));
}
