cmake_minimum_required(VERSION 3.4...3.18)

project(constrainedGDlib LANGUAGES CXX)


file(GLOB_RECURSE PROJECT_SOURCE 
	"src/*.h"
	"src/*.cpp"
	"../constrainedGD/src/*.h"
	"../constrainedGD/src/*.cpp")

include_directories(src)
include_directories(../constrainedGD/src)

if (MSVC)
    # warning level 4 and all warnings as errors
    add_compile_options()
else()
    # lots of warnings and all warnings as errors
    add_compile_options("-Ofast")
endif()
	
add_subdirectory(../../../pybind11-master pybind11-master) #path to pybind

pybind11_add_module(constrainedGDlib ${PROJECT_SOURCE})
