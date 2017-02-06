PROJECT(latch_cuda)

FIND_PACKAGE(CUDA REQUIRED)
FIND_PACKAGE(OpenCV 3 REQUIRED)

FILE(GLOB 
    SOURCES "LatchClassifier.cpp" "LatchClassifierCV.cpp" "LatchBitMatcher.cpp" "*.c" "*.h" ".hpp")

INCLUDE(FindCUDA)

SET(CUDASRCS
    latch.cu
    bitMatcher.cu
)

LIST(APPEND CMAKE_CXX_FLAGS "-std=c++11 -O3")
LIST(APPEND CMAKE_C_FLAGS "-std=c++11 -O3")
LIST(APPEND CUDA_NVCC_FLAGS "-D_MWAITXINTRIN_H_INCLUDED -lineinfo -Xptxas -v -Xcompiler -fopenmp -use_fast_math -O3 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_60,code=sm_60 --default-stream per-thread")

MESSAGE("CUDA_NVCC_FLAGS: ${CUDA_NVCC_FLAGS}")

SET(CUDA_PROPAGATE_HOST_FLAGS OFF)

INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

CUDA_ADD_LIBRARY(latch_cuda ${CUDASRCS} ${SOURCES} SHARED)

TARGET_LINK_LIBRARIES(latch_cuda ${OpenCV_LIBS})

ADD_DEFINITIONS(-DTHIS_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../image/")
