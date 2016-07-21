#include "LatchClassifierOpenMVG.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits>
#include <iostream>
#include "opencv2/core/mat.hpp"
#include "opencv2/core/eigen.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/core/cuda_stream_accessor.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/xfeatures2d.hpp"

#include "bitMatcher.h"
#include "latch.h"

/* Helper functions. */

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
	m_count = 0;
}

bool compareFunction(cv::KeyPoint p1, cv::KeyPoint p2) {return p1.response>p2.response;}
//The function retains the stongest M keypoints in kp
void RetainBestKeypoints(std::vector<cv::KeyPoint>  &kp, int M)
{
	if (kp.size() < M)
		int gil=1;

	sort(kp.begin(),kp.end(),compareFunction);
	if (kp.size()>M)
		kp.erase(kp.begin()+M,kp.end());

}

std::vector<LatchClassifierKeypoint> LatchClassifierOpenMVG::identifyFeaturePointsOpenMVG(Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> img) {
    cv::Mat imgConverted;
    cv::eigen2cv(img, imgConverted);
    if (m_width != imgConverted.size().width || m_height != imgConverted.size().height) {
        setImageSize(imgConverted.size().width, imgConverted.size().height);
        m_width = imgConverted.size().width;
        m_height = imgConverted.size().height;
    }
		// 1. Find all keypoints
		cv::Ptr<cv::BRISK> briskDetector = cv::BRISK::create();
		std::vector<cv::KeyPoint> briefKeypoints;
		briskDetector->detect(imgConverted, briefKeypoints, cv::noArray());
		briskDetector.release();
		const size_t briskSize = briefKeypoints.size();
		std::cout << "Got brisk size: " << briskSize << std::endl;

		// 2. Use BRIEF size to feed into SIFT
		cv::Ptr<cv::xfeatures2d::SIFT> siftDetector = cv::xfeatures2d::SIFT::create(briskSize);
		std::vector<cv::KeyPoint> siftKeypoints;
		siftDetector->detect(imgConverted, siftKeypoints, cv::noArray());
		RetainBestKeypoints(siftKeypoints, briskSize);
		siftDetector.release();
		
		std::cout << "Got siftDetector. Size: " << siftKeypoints.size() << "vs brisk: " << briskSize << std::endl;
		
		std::vector<cv::KeyPoint> returnedKeypoints(siftKeypoints.begin(), siftKeypoints.begin() + m_testingArr[m_count]);
		// 3. Run CLATCH across all keypoint descriptors
 		
    // Convert image to grayscale
    cv::cuda::GpuMat img1g;
		{
      cv::cuda::GpuMat imgGpu;
      imgGpu.upload(imgConverted, m_stream);

      imgConverted.channels() == 3 ? cv::cuda::cvtColor(imgGpu, img1g, CV_BGR2GRAY, 0, m_stream) : img1g.upload(imgConverted, m_stream);
    }
    
		cudaStream_t copiedStream = cv::cuda::StreamAccessor::getStream(m_stream);
 		
		{
      cv::cuda::GpuMat d_keypoints;
	
      m_orbClassifier->detectAsync(img1g, d_keypoints, cv::noArray(), m_stream);
      cudaStreamSynchronize(copiedStream);
	
		  m_orbClassifier->convert(d_keypoints, returnedKeypoints);
			// This MUST be a multiple of 16 to work with CLATCH.
			returnedKeypoints.resize(returnedKeypoints.size() - (returnedKeypoints.size() % 16));
		}

		int numKP0;
		std::cout << "Running latch" << std::endl;
    latch(imgConverted, m_dI, m_pitch, m_hK1, m_dD1, &numKP0, m_maxKP, m_dK, &returnedKeypoints, m_dMask, copiedStream, m_latchFinished);
 		std::cout << "Ran latch" << std::endl;
    size_t sizeD = m_maxKP * (2048 / 32) * sizeof(unsigned int); // D for descriptor
		std::cout << "Copying memory back" << std::endl;
    cudaMemcpyAsync(m_hD1, m_dD1, sizeD, cudaMemcpyDeviceToHost, copiedStream);
		std::cout << "Memory copied" << std::endl;
    m_stream.waitForCompletion();

		m_count++;
    return convertCVKeypointsToCustom(returnedKeypoints);
}

unsigned int* LatchClassifierOpenMVG::describeOpenMVG(Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> img, 
		std::vector<cv::KeyPoint>& keypoints) {
    cv::Mat imgConverted;
    cv::eigen2cv(img, imgConverted);
    if (m_width != imgConverted.size().width || m_height != imgConverted.size().height) {
        setImageSize(imgConverted.size().width, imgConverted.size().height);
        m_width = imgConverted.size().width;
        m_height = imgConverted.size().height;
    }

    // Convert image to grayscale
    cv::cuda::GpuMat img1g;
		{
      cv::cuda::GpuMat imgGpu;
      imgGpu.upload(imgConverted, m_stream);

      imgConverted.channels() == 3 ? cv::cuda::cvtColor(imgGpu, img1g, CV_BGR2GRAY, 0, m_stream) : img1g.upload(imgConverted, m_stream);
    }
    cudaStream_t copiedStream = cv::cuda::StreamAccessor::getStream(m_stream);
    int numKP0;
    latch(imgConverted, m_dI, m_pitch, m_hK1, m_dD1, &numKP0, m_maxKP, m_dK, &keypoints, m_dMask, copiedStream, m_latchFinished);
    
    size_t sizeD = m_maxKP * (2048 / 32) * sizeof(unsigned int); // D for descriptor
    cudaMemcpyAsync(m_hD1, m_dD1, sizeD, cudaMemcpyDeviceToHost, copiedStream);

    m_stream.waitForCompletion();

		return m_hD1;
}

LatchClassifierOpenMVG::~LatchClassifierOpenMVG() {

}
