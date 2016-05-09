PROJECT(latch_cuda)

FIND_PACKAGE(CUDA REQUIRED)

FILE(GLOB 
    SOURCES "*.cu" "*.cpp" "*.c" "*.h")

INCLUDE(FindCUDA)

INCLUDE_DIRECTORIES(/opt/cuda/include /usr/local/include)

SET(CUDASRCS
    latch.cu
    bitMatcher.cu
    LatchClassifier.cpp
    min.cpp
)

LIST(APPEND CMAKE_CXX_FLAGS "-std=c++11 -O3 -I/opt/cuda/include -L/opt/cuda/lib64 -lcuda -lcudart -gencode
arch=compute_30,code=sm_30")
LIST(APPEND CMAKE_C_FLAGS "-std=c++11 -O3 -I/opt/cuda/include -L/opt/cuda/lib64 -lcuda -lcudart")
set (CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11; --compiler-options -Wall -g -arch compute_30 -code sm_30")
message("CUDA_NVCC_FLAGS: ${CUDA_NVCC_FLAGS}")
LIST(APPEND CUDA_NVCC_FLAGS -lineinfo -use_fast_math -Xptxas)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)
LIST(APPEND CUDA_NVCC_FLAGS --default-stream per-thread)
LIST(APPEND CUDA_NVCC_FLAGS -I/usr/local/include -L/usr/local/lib -I/opt/cuda/include -L/opt/cuda/lib64 -lcuda
-lcudart)

cuda_compile(LATCH_O latch.cu OPTIONS CUDA_NVCC_FLAGS)
cuda_compile(BIN_O binMatcher.cu OPTIONS CUDA_NVCC_FLAGS)

add_definitions(-DTHIS_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../image/")
