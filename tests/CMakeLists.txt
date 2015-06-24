find_package(Boost COMPONENTS unit_test_framework REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

file(GLOB TEST_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)

foreach(test_src ${TEST_SRCS})
        get_filename_component(test_name ${test_src} NAME_WE)
        add_executable(${test_name} ${test_src})
        target_link_libraries(${test_name} ${Boost_LIBRARIES} ppfmap)
        set_target_properties(${test_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY 
            ${CMAKE_CURRENT_BINARY_DIR}/test_bin)
        add_test(NAME ${test_name} 
                 WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/test_bin
                 COMMAND ${CMAKE_CURRENT_BINARY_DIR}/test_bin/${test_name})
endforeach(test_src)
