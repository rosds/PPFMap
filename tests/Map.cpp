#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "PPF Map test module"

#include <boost/test/included/unit_test.hpp>
#include <PPFMap/ppf_cuda_calls.h>


BOOST_AUTO_TEST_CASE(constructors) {

    /** Create simple pointcloud example **/
    float3 points[3];
    points[1] = make_float3(-1.0f, 0.0f, 0.0f);
    points[2] = make_float3(0.0f, sqrtf(3.0f), 0.0f);
    points[3] = make_float3(1.0f, 0.0f, 0.0f);

    float3 normals[3];
    normals[1] = make_float3(-0.5f * sqrtf(3.0f), -0.5f, 0.0f);
    normals[2] = make_float3(0.0f, 1.0f, 0.0f);
    normals[3] = make_float3(0.5f * sqrtf(3.0f), -0.5f, 0.0f);

    ppfmap::Map::Ptr map1 = ppfmap::cuda::setPPFMap(points, normals, 3, 1.0f, 1.0f);
    BOOST_CHECK_EQUAL(map1->size(), std::size_t(9));
}
