PROJECT(latch_cuda)

FIND_PACKAGE(CUDA REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED)

FILE(GLOB 
    SOURCES "LatchClassifier.cpp" "LatchClassifierOpenMVG.cpp" "LatchClassifierCV.cpp" "LatchBitMatcher.cpp" "*.c" "*.h" ".hpp")

INCLUDE(FindCUDA)

SET(CUDASRCS
    latch.cu
    bitMatcher.cu
)

#LIST(APPEND CMAKE_CXX_FLAGS "-std=c++11 -O3 -I/opt/cuda/include -L/opt/cuda/lib64 -lcuda -lcudart -gencode arch=compute_30,code=sm_30")
#LIST(APPEND CMAKE_C_FLAGS "-std=c++11 -O3 -I/opt/cuda/include -L/opt/cuda/lib64 -lcuda -lcudart")
LIST(APPEND CUDA_NVCC_FLAGS "-D_MWAITXINTRIN_H_INCLUDED -lineinfo -Xptxas -v -Xcompiler -fopenmp -use_fast_math -O3 -gencode arch=compute_60,code=sm_60 --default-stream per-thread")
message("CUDA_NVCC_FLAGS: ${CUDA_NVCC_FLAGS}")

SET(CUDA_PROPAGATE_HOST_FLAGS OFF)

cuda_add_library(latch_cuda ${CUDASRCS} ${SOURCES} SHARED)

install(TARGETS latch_cuda DESTINATION lib EXPORT openMVG-targets)

target_link_libraries(latch_cuda ${OpenCV_LIBS})

add_definitions(-DTHIS_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../image/")
