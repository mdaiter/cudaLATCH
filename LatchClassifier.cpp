#include "LatchClassifier.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits>
#include <iostream>
#include "opencv2/core/cuda.hpp"
#include "opencv2/core/cuda_stream_accessor.hpp"
#include "opencv2/cudaimgproc.hpp"

#include "bitMatcher.h"
#include "latch.h"

/* Helper functions. */

#define cudaCalloc(A, B) \
    do { \
        cudaError_t __cudaCalloc_err = cudaMalloc(A, B); \
        if (__cudaCalloc_err == cudaSuccess) cudaMemset(*A, 0, B); \
    } while (0)

#define checkError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define checkLaunchError()                                            \
do {                                                                  \
    /* Check synchronous errors, i.e. pre-launch */                   \
    cudaError_t err = cudaGetLastError();                             \
    if (cudaSuccess != err) {                                         \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cudaGetErrorString(err) );       \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
    /* Check asynchronous errors, i.e. kernel failed (ULF) */         \
    err = cudaThreadSynchronize();                                    \
    if (cudaSuccess != err) {                                         \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cudaGetErrorString( err) );      \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
} while (0)


/* Main class definition */

LatchClassifier::LatchClassifier() :
    m_maxKP(512 * NUM_SM),
    m_matchThreshold(12),
    m_detectorThreshold(10),
    m_detectorTargetKP(3000),
    m_detectorTolerance(200),
    m_shouldBeTimed(false),
    m_defects(0.0),
    m_stream(cv::cuda::Stream::Null()),
    m_stream1(cv::cuda::Stream::Null()),
    m_stream2(cv::cuda::Stream::Null()) {
	size_t sizeK = m_maxKP * sizeof(float) * 4; // K for keypoints
	size_t sizeD = m_maxKP * (2048 / 32) * sizeof(unsigned int); // D for descriptor
	size_t sizeM = m_maxKP * sizeof(int); // M for Matches

    cudaMallocHost((void**) &m_hK1, sizeK);
    cudaMallocHost((void**) &m_hK2, sizeK);
    cudaMallocHost((void**) &m_hD1, sizeD);
    cudaMallocHost((void**) &m_hD2, sizeD);

    cudaCalloc((void**) &m_dK, sizeK);
    cudaCalloc((void**) &m_dD1, sizeD);
    cudaCalloc((void**) &m_dD2, sizeD);
    cudaCalloc((void**) &m_dM1, sizeM);
    cudaCalloc((void**) &m_dM2, sizeM);
    cudaCalloc((void**) &m_dMask, sizeM);
    cudaEventCreate(&m_latchFinished);

    // The patch triplet locations for LATCH fits in memory cache.
    loadPatchTriplets(m_patchTriplets);

    float h_mask[64];
    for (size_t i = 0; i < 64; i++) { h_mask[i] = 1.0f; }
    initMask(&m_dMask, h_mask);

    m_orbClassifier = cv::cuda::ORB::create();
    m_orbClassifier->setBlurForDescriptor(true);

    m_orbClassifierCPU = cv::ORB::create(10000);
    m_latch = cv::xfeatures2d::LATCH::create();
}

void LatchClassifier::setImageSize(int width, int height) {
    size_t sizeI = width * height * sizeof(unsigned char);
    cudaCalloc((void**) &m_dI, sizeI);
    initImage(&m_dI, width, height, &m_pitch);
    std::cout << "Finished setting image size: " << width << " " << height << std::endl;
}

LatchClassifier::~LatchClassifier() {
    cudaStreamDestroy(cv::cuda::StreamAccessor::getStream(m_stream));
    cudaStreamDestroy(cv::cuda::StreamAccessor::getStream(m_stream1));
    cudaStreamDestroy(cv::cuda::StreamAccessor::getStream(m_stream2));
    cudaFreeArray(m_patchTriplets);
    cudaFree(m_dK);
    cudaFree(m_dD1);
    cudaFree(m_dD2);
    cudaFree(m_dM1);
    cudaFree(m_dM2);
    cudaFreeHost(m_hK1);
    cudaFreeHost(m_hK2);
}
