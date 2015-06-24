#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "PPF Map test module"

#include <boost/test/included/unit_test.hpp>
#include <PPFMap/Map.h>


BOOST_AUTO_TEST_CASE(constructors) {

    /** test empty constructor **/
    ppfmap::Map map1;
    BOOST_CHECK_EQUAL(map1.size(), std::size_t(0));
}
