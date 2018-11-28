##find cuda package and set opencl directories (included in cuda installation)
#find_package(CUDA)
#not finding the toolkit root dir. ATM hardcoding its direcory path
SET(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.0")
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)

file(GLOB_RECURSE SOURCES *.cpp *.h *.c)
add_executable(cw2_jpeg_compression ${SOURCES})

target_link_libraries(cw2_jpeg_compression ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/OpenCL.lib)

# do the copying
#foreach( file_i ${THIRD_PARTY_DLLS})
#    add_custom_command(
#    TARGET cw2_jpeg_compression
#    POST_BUILD
#    COMMAND ${CMAKE_COMMAND}
#    ARGS -E copy ${CMAKE_SOURCE_DIR}/source/jpeg-compressor/test.cl ${CMAKE_CURRENT_BINARY_DIR}/test.cl
#)
#endforeach( file_i )