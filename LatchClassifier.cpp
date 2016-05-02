#include "LatchClassifier.h"

#include <iostream>
#include "opencv2/core/cuda.hpp"

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
    m_defects(0.0) {
	size_t sizeK = m_maxKP * sizeof(float) * 4; // K for keypoints
	size_t sizeD = m_maxKP * (2048 / 32) * sizeof(unsigned int); // D for descriptor
	size_t sizeM = m_maxKP * sizeof(int); // M for Matches

    cudaMallocHost((void**) &m_hK1, sizeK);
    cudaMallocHost((void**) &m_hK2, sizeK);

    cudaCalloc((void**) &m_dK, sizeK);
    cudaCalloc((void**) &m_dD1, sizeD);
    cudaCalloc((void**) &m_dD2, sizeD);
    cudaCalloc((void**) &m_dM1, sizeM);
    cudaCalloc((void**) &m_dM2, sizeM);
    cudaCalloc((void**) &m_dMask, sizeM);
    cudaEventCreate(&m_latchFinished);

    // The patch triplet locations for LATCH fits in memory cache.
    loadPatchTriplets(m_patchTriplets);

    cudaStreamCreate(&m_stream1);
    cudaStreamCreate(&m_stream2);

    float h_mask[64];
    for (size_t i = 0; i < 64; i++) { h_mask[i] = 1.0f; }
    initMask(&m_dMask, h_mask);
}

void LatchClassifier::setImageSize(int width, int height) {
    size_t sizeI = width * height * sizeof(unsigned char);
    cudaCalloc((void**) &m_dI, sizeI);
    initImage(&m_dI, width, height, &m_pitch);
    std::cout << "Finished setting image size: " << width << " " << height << std::endl;
}

std::vector<cv::KeyPoint> LatchClassifier::identifyFeaturePoints(cv::Mat& img) {

    // Convert image to grayscale
    cv::Mat img1g;
    
    cv::cvtColor(img, img1g, CV_BGR2GRAY);
    // Find features using ORB/FAST
    std::vector<cv::KeyPoint> keypoints;
    FAST(img1g, keypoints, m_detectorThreshold);

    int numKP0;
    latch(img1g, m_dI, m_pitch, m_hK1, m_dD1, &numKP0, m_maxKP, m_dK, &keypoints, m_dMask, m_latchFinished);
    
    return keypoints;
}

std::tuple<std::vector<cv::KeyPoint>, 
            std::vector<cv::KeyPoint>, 
            std::vector<cv::DMatch>>
            LatchClassifier::identifyFeaturePointsBetweenImages(cv::Mat& img1, cv::Mat& img2) {
    std::vector<cv::KeyPoint> goodMatches1;
    std::vector<cv::KeyPoint> goodMatches2;
    std::vector<cv::DMatch> goodMatches3;
    // Images MUST match each other in width and height.
    if (img2.cols != img1.cols || img2.rows != img2.rows)
        return std::make_tuple(goodMatches1, goodMatches2, goodMatches3);

    // Convert image to grayscale
    cv::Mat img1g;
    cv::Mat img2g;
    
    cv::cvtColor(img1, img1g, CV_BGR2GRAY);
    cv::cvtColor(img2, img2g, CV_BGR2GRAY);
    // Find features using ORB/FAST
    std::vector<cv::KeyPoint> keypoints0;
    // cv::gpu::FAST_GPU gpuFast(10, true);
    FAST(img1g, keypoints0, m_detectorThreshold);
    // gpuFast(img1g, cv::gpu::GpuMat() keypoints0);

    int numKP0;
    latch(img1g, m_dI, m_pitch, m_hK1, m_dD1, &numKP0, m_maxKP, m_dK, &keypoints0, m_dMask, m_latchFinished);

    std::vector<cv::KeyPoint> keypoints1;
    // gpuFast(img2g, cv::gpu::GpuMat(), keypoints1);
    FAST(img2g, keypoints1, m_detectorThreshold);
    int numKP1;
    latch(img2g, m_dI, m_pitch, m_hK2, m_dD2, &numKP1, m_maxKP, m_dK, &keypoints1, m_dMask, m_latchFinished);

    bitMatcher(m_dD1, m_dD2, numKP0, numKP1, m_maxKP, m_dM1, m_matchThreshold, m_stream1, m_latchFinished);
    bitMatcher(m_dD2, m_dD1, numKP1, numKP0, m_maxKP, m_dM2, m_matchThreshold, m_stream2, m_latchFinished);

    // Recombine to find intersecting features. Need to declare arrays as static due to size.
//    int* h_M1, *h_M2;
//    cudaMallocHost((void**) &h_M1, sizeof(int) * m_maxKP);
//    cudaMallocHost((void**) &h_M2, sizeof(int) * m_maxKP);
    int h_M1[m_maxKP];
    int h_M2[m_maxKP];
    getMatches(m_maxKP, h_M1, m_dM1);
    getMatches(m_maxKP, h_M2, m_dM2);
    
    for (size_t i = 0; i < numKP0; i++) {
        if (h_M1[i] >= 0 && h_M1[i] < numKP1 && h_M2[h_M1[i]] == i) {
            goodMatches1.push_back(keypoints0[i]);
            goodMatches2.push_back(keypoints1[h_M1[i]]);
            goodMatches3.push_back(cv::DMatch(i, h_M1[i], 0));
        }
    }

//    cudaFree(h_M1);
//    cudaFree(h_M2);

    return std::make_tuple(goodMatches1, goodMatches2, goodMatches3);
}

LatchClassifier::~LatchClassifier() {
    cudaStreamDestroy(m_stream1);
    cudaStreamDestroy(m_stream2);
    cudaFreeArray(m_patchTriplets);
    cudaFree(m_dK);
    cudaFree(m_dD1);
    cudaFree(m_dD2);
    cudaFree(m_dM1);
    cudaFree(m_dM2);
    cudaFreeHost(m_hK1);
    cudaFreeHost(m_hK2);
}
