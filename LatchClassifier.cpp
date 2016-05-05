#include "LatchClassifier.h"

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
    m_latch = xfeatures2d::LATCH::create();
}

void LatchClassifier::setImageSize(int width, int height) {
    size_t sizeI = width * height * sizeof(unsigned char);
    cudaCalloc((void**) &m_dI, sizeI);
    initImage(&m_dI, width, height, &m_pitch);
    std::cout << "Finished setting image size: " << width << " " << height << std::endl;
}

std::vector<cv::KeyPoint> LatchClassifier::identifyFeaturePointsCPU(cv::Mat& img) {
    // Convert image to grayscale
    cv::Mat img1g;
 
    cv::cvtColor(img, img1g, CV_BGR2GRAY);
    // Find features using ORB/FAST
    std::vector<cv::KeyPoint> keypoints;
    
    m_orbClassifierCPU->detect(img1g, keypoints);
    Mat desc;
    m_latch->compute(img1g, keypoints, desc);

    return keypoints;
}

void LatchClassifier::writeSIFTFile(const std::string& filename, int width, int height, unsigned int* desc, std::vector<cv::KeyPoint>& keys) {
   	FILE* f = fopen(filename.c_str(), "wb");

    fprintf(f, "%d %d \n", width, height);
    int count = 0;
    for (int i = 0; i < keys.size() * 64; i++) {
        std::cout << "Desc[" << i << "]: " << static_cast<float>(static_cast<long>(desc[i])) << std::endl;
        if (desc[i]) count++;
    }
//    std::cout << "Count: " << count << std::endl;
//    std::cout << "Size: " << keys.size() << std::endl;

    // Normalize descriptor
    for (int i = 0; i < keys.size() ; i++) {
        float norm = 0.0;
        for (int j = 0; j < 64; j++) {
            int index = i * 64 + j;
            union {
                unsigned int ui;
                float f;
            } conversion_union = { .ui = desc[index] };
            float toBeAdded = conversion_union.f * conversion_union.f;
            if (toBeAdded < std::numeric_limits<float>::max()) 
                norm += toBeAdded;
            else {
                norm = std::numeric_limits<float>::max();
                break;
            }
        }
        norm = 512.0/std::max(std::sqrt(norm),1.19209290E-07F);
//        std::cout << "Norm: " << norm << std::endl;
        for (int j = 0; j < 64; j++) {
            int index = i * 64 + j;
            float tempFloat = static_cast<float>(static_cast<long>(desc[index])) * norm;
//            std::cout << "New unsigned float : " << tempFloat << std::endl;
        }

        count++;

      	fprintf(f, "%f %f %f %f \n", keys.at(i).pt.y, keys.at(i).pt.x, keys.at(i).size, (keys.at(i).angle*M_PI/180.0));
       	for ( int j = 0; j < 16; j++) {
            unsigned int tempInt = desc[i * 64 + j];
            float tempFloat = static_cast<float>(static_cast<long>(desc[i * 64 + j])) * norm;
            unsigned char* x = reinterpret_cast<unsigned char*>(&tempFloat);
            for (int k = 0; k < 4; k++) {
                //int bitShiftVal = 32 - 8 * (k+1);
                //unsigned char x = (tempFloat >> bitShiftVal) & 0xFF;
                fprintf(f, "%u ", x[k]);
       		    if ((j * 4 + k + 1) % 19 == 0) fprintf(f, "\n");
            }
            count++;
       	}
        for ( int j = 64; j < 128; j++) {
            fprintf(f, "0 ");
            if ((j + 1) % 19 == 0) fprintf(f, "\n");
        }
       	fprintf(f, "\n");
    }
//    std::cout << "Count: " << count << std::endl;

    fclose(f);

}

std::vector<cv::KeyPoint> LatchClassifier::identifyFeaturePoints(cv::Mat& img) {
    cv::cuda::GpuMat imgGpu;
    imgGpu.upload(img, m_stream);

    // Convert image to grayscale
    cv::cuda::GpuMat img1g;
 
    cv::cuda::cvtColor(imgGpu, img1g, CV_BGR2GRAY, 0, m_stream);
    // Find features using ORB/FAST
    std::vector<cv::KeyPoint> keypoints;
    cuda::GpuMat d_keypoints;
    m_orbClassifier->detectAsync(img1g, d_keypoints, cv::noArray(), m_stream);
    cudaStream_t copiedStream = cv::cuda::StreamAccessor::getStream(m_stream);
    cudaStreamSynchronize(copiedStream);
    m_orbClassifier->convert(d_keypoints, keypoints);

    int numKP0;
    latchGPU(img1g, m_pitch, m_hK1, m_dD1, &numKP0, m_maxKP, m_dK, &keypoints, m_dMask, copiedStream, m_latchFinished);
	
    size_t sizeD = m_maxKP * (2048 / 32) * sizeof(unsigned int); // D for descriptor
    cudaMemcpyAsync(m_hD1, m_dD1, sizeD, cudaMemcpyDeviceToHost, copiedStream);
    
    m_stream.waitForCompletion();
    return keypoints;
}

void LatchClassifier::identifyFeaturePointsAsync(cv::Mat& img, 
                                                 cv::cuda::Stream::StreamCallback callback, 
                                                  void* userData) {
    cv::cuda::Stream& stream = cv::cuda::Stream::Null();
    cv::cuda::GpuMat imgGpu;
    imgGpu.upload(img, stream);

    // Convert image to grayscale
    cv::cuda::GpuMat img1g;
 
    cv::cuda::cvtColor(imgGpu, img1g, CV_BGR2GRAY, 0, stream);
    // Find features using ORB/FAST
    std::vector<cv::KeyPoint> keypoints;
    cuda::GpuMat d_keypoints;
    m_orbClassifier->detectAsync(img1g, d_keypoints, cv::noArray(), stream);
    cudaStream_t copiedStream = cv::cuda::StreamAccessor::getStream(stream);
    cudaStreamSynchronize(copiedStream);
    m_orbClassifier->convert(d_keypoints, keypoints);

    int numKP0;
    latchGPU(img1g, m_pitch, m_hK1, m_dD1, &numKP0, m_maxKP, m_dK, &keypoints, m_dMask, copiedStream, m_latchFinished);
    
    size_t sizeD = m_maxKP * (2048 / 32) * sizeof(unsigned int); // D for descriptor
    cudaMemcpyAsync(m_hD1, m_dD1, sizeD, cudaMemcpyDeviceToHost, copiedStream);
    
    stream.enqueueHostCallback(callback, userData);
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

    cv::cuda::GpuMat imgGpu1;
    cv::cuda::GpuMat imgGpu2;

    imgGpu1.upload(img1, m_stream1);
    imgGpu2.upload(img2, m_stream2);

    // Convert image to grayscale
    cv::cuda::GpuMat img1g;
    cv::cuda::GpuMat img2g;
    
    cv::cuda::cvtColor(imgGpu1, img1g, CV_BGR2GRAY, 0, m_stream1);
    cv::cuda::cvtColor(imgGpu2, img2g, CV_BGR2GRAY, 0, m_stream2);
    // Find features using ORB/FAST
    std::vector<cv::KeyPoint> keypoints0;
    std::vector<cv::KeyPoint> keypoints1;
    
    cv::cuda::GpuMat d_keypoints0;
    cv::cuda::GpuMat d_keypoints1;
    
    m_orbClassifier->detectAsync(img1g, d_keypoints0, cv::noArray(), m_stream1);
    m_orbClassifier->detectAsync(img2g, d_keypoints1, cv::noArray(), m_stream2);
    
    m_stream1.waitForCompletion();
    m_stream2.waitForCompletion();
    
    m_orbClassifier->convert(d_keypoints0, keypoints0);
    m_orbClassifier->convert(d_keypoints1, keypoints1);

    cudaStream_t copiedStream1 = cv::cuda::StreamAccessor::getStream(m_stream1);
    cudaStream_t copiedStream2 = cv::cuda::StreamAccessor::getStream(m_stream2);

    int numKP0;
    latchGPU(img1g, m_pitch, m_hK1, m_dD1, &numKP0, m_maxKP, m_dK, &keypoints0, m_dMask, copiedStream1, m_latchFinished);

    int numKP1;
    latchGPU(img2g, m_pitch, m_hK2, m_dD2, &numKP1, m_maxKP, m_dK, &keypoints1, m_dMask, copiedStream2, m_latchFinished);

    size_t sizeD = m_maxKP * (2048 / 32) * sizeof(unsigned int); // D for descriptor
    cudaMemcpyAsync(m_hD1, m_dD1, sizeD, cudaMemcpyDeviceToHost, copiedStream1);
    cudaMemcpyAsync(m_hD2, m_dD2, sizeD, cudaMemcpyDeviceToHost, copiedStream2);
    
    bitMatcher(m_dD1, m_dD2, numKP0, numKP1, m_maxKP, m_dM1, m_matchThreshold, copiedStream1, m_latchFinished);
    bitMatcher(m_dD2, m_dD1, numKP1, numKP0, m_maxKP, m_dM2, m_matchThreshold, copiedStream2, m_latchFinished);

    cudaStreamSynchronize(copiedStream1);
    cudaStreamSynchronize(copiedStream2);

    // Recombine to find intersecting features. Need to declare arrays as static due to size.
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

    return std::make_tuple(goodMatches1, goodMatches2, goodMatches3);
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
