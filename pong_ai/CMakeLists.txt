cmake_minimum_required(VERSION 3.15)
project(PongAITests)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(include)

# Find all test files
file(GLOB TEST_SOURCES "src/test/*.cpp")

# Create an executable for each test
foreach(test_source ${TEST_SOURCES})
    get_filename_component(test_name ${test_source} NAME_WE)
    add_executable(${test_name} ${test_source})
    target_link_libraries(${test_name} Threads::Threads)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()