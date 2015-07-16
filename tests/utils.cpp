#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "Utility functions tests"

#include <boost/test/included/unit_test.hpp>
#include <PPFMap/utils.h>


BOOST_AUTO_TEST_CASE(dot_function) {
    float3 x = make_float3(1.0f, 0.0f, 0.0f);
    float3 y = make_float3(0.0f, 1.0f, 0.0f);
    float3 z = make_float3(0.0f, 0.0f, 1.0f);

    /** Check dot product function **/
    BOOST_CHECK_EQUAL(ppfmap::dot(x, y), 0.0f);
    BOOST_CHECK_EQUAL(ppfmap::dot(x, z), 0.0f);
    BOOST_CHECK_EQUAL(ppfmap::dot(y, z), 0.0f);
    BOOST_CHECK_EQUAL(ppfmap::dot(x, x), 1.0f);
    BOOST_CHECK_EQUAL(ppfmap::dot(y, y), 1.0f);
    BOOST_CHECK_EQUAL(ppfmap::dot(z, z), 1.0f);
}

BOOST_AUTO_TEST_CASE(norm_function) {
    float3 x = make_float3(1.0f, 0.0f, 0.0f);
    float3 y = make_float3(0.0f, 1.0f, 0.0f);
    float3 z = make_float3(0.0f, 0.0f, 1.0f);

    /** Check norm function **/
    BOOST_CHECK_EQUAL(ppfmap::norm(x), 1.0f);
    BOOST_CHECK_EQUAL(ppfmap::norm(y), 1.0f);
    BOOST_CHECK_EQUAL(ppfmap::norm(z), 1.0f);
}
