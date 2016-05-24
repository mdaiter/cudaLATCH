#include "LatchClassifierOpenMVG.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits>
#include <iostream>
#include "opencv2/core/mat.hpp"
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

LatchClassifierOpenMVG::LatchClassifierOpenMVG() :
    LatchClassifier() {
}

std::vector<LatchClassifierKeypoint> convertCVKeypointsToCustom(std::vector<cv::KeyPoint>& keypointsCV) {
    std::vector<LatchClassifierKeypoint> keypoints;
    for (size_t i = 0; i < keypointsCV.size(); i++) {
        LatchClassifierKeypoint kp(
            keypointsCV[i].pt.x,
            keypointsCV[i].pt.y,
            keypointsCV[i].angle,
            keypointsCV[i].size
        );
        keypoints.push_back(kp);
    }
    return keypoints;
}

std::vector<LatchClassifierKeypoint> LatchClassifierOpenMVG::identifyFeaturePointsOpenMVG(Eigen::Matrix<unsigned char, -1, -1, 1 , -1, -1> img) {
    cv::Mat imgConverted;
    cv::eigen2cv(img, imgConverted);
    cv::cuda::GpuMat imgGpu;
    imgGpu.upload(imgConverted, m_stream);

    // Convert image to grayscale
    cv::cuda::GpuMat img1g;

    imgConverted.channels() == 3 ? cv::cuda::cvtColor(imgGpu, img1g, CV_BGR2GRAY, 0, m_stream) : img1g.upload(imgConverted, m_stream);

    // Find features using ORB/FAST
    std::vector<cv::KeyPoint> keypoints;
    cv::cuda::GpuMat d_keypoints;
    m_orbClassifier->detectAsync(img1g, d_keypoints, cv::noArray(), m_stream);
    cudaStream_t copiedStream = cv::cuda::StreamAccessor::getStream(m_stream);
    cudaStreamSynchronize(copiedStream);
    m_orbClassifier->convert(d_keypoints, keypoints);

    int numKP0;
    latchGPU(img1g, m_pitch, m_hK1, m_dD1, &numKP0, m_maxKP, m_dK, &keypoints, m_dMask, copiedStream, m_latchFinished);
	
    size_t sizeD = m_maxKP * (2048 / 32) * sizeof(unsigned int); // D for descriptor
    cudaMemcpyAsync(m_hD1, m_dD1, sizeD, cudaMemcpyDeviceToHost, copiedStream);
    
    m_stream.waitForCompletion();

    return convertCVKeypointsToCustom(keypoints);
}

LatchClassifierOpenMVG::~LatchClassifierOpenMVG() {
}
