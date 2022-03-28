cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(example-app)



set( CMAKE_CXX_FLAGS "-std=c++14 -Wall" )
set( CMAKE_CXX_FLAGS_DEBUG "-g -std=c++14 -Wall" )
message( "Mac settings chosen..." )
include_directories( include )


set(Torch_DIR ${CMAKE_SOURCE_DIR}/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(OpenCV_DIR /Users/yasinmac/Desktop/yasin_/github/c-/opencv/install/lib/cmake/opencv4)
find_package(OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})


# set(ncnn_DIR /home/syn1/Desktop/OTHERS/yasin_github/c-/include/ncnn-20210525-ubuntu-2004/lib/cmake/ncnn)
find_package(ncnn REQUIRED)


file ( GLOB SOURCES "./src/*.cpp" "./include/*.h" )

add_executable( ${PROJECT_NAME} ${SOURCES} )

target_link_libraries(example-app ncnn ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set_property(TARGET example-app PROPERTY CXX_STANDARD 14)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET example-app
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:example-app>)
endif (MSVC)