#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "PPF feature class test"

#include <boost/test/included/unit_test.hpp>
#include <PPFMap/ppf.h>


BOOST_AUTO_TEST_CASE(constructors) {
    /** Empty constructor **/
    ppfmap::PPFFeature a;
    BOOST_CHECK_EQUAL(a.f1, 0.0f);
    BOOST_CHECK_EQUAL(a.f2, 0.0f);
    BOOST_CHECK_EQUAL(a.f3, 0.0f);
    BOOST_CHECK_EQUAL(a.f4, 0.0f);

    float3 p1 = make_float3(0.0f, 0.0f, 0.0f);
    float3 p2 = make_float3(1.0f, 0.0f, 0.0f);
    float3 n1 = make_float3(0.0f, 1.0f, 0.0f);
    ppfmap::PPFFeature b(p1, n1, p2, n1);
    BOOST_CHECK_EQUAL(b.f1, 1.0f);
    BOOST_CHECK_EQUAL(b.f2, 0.5f * static_cast<float>(M_PI));
    BOOST_CHECK_EQUAL(b.f3, 0.5f * static_cast<float>(M_PI));
    BOOST_CHECK_EQUAL(b.f4, 0.0f);
}
