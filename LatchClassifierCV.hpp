#ifndef LATCH_CLASSIFIER_CV_H
#define LATCH_CLASSIFIER_CV_H

#include <tuple>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include "opencv2/core/mat.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/cudafeatures2d.hpp"
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/imgcodecs.hpp>

#include "LatchBitMatcherMatch.hpp"
#include "LatchClassifier.hpp"
#include "LatchClassifierKeypoint.hpp"
#include "params.hpp"

class LatchClassifierCV : public LatchClassifier {
    public:
        LatchClassifierCV();
        // This *must* be called before identifyFeaturePoints is called
        void identifyFeaturePointsAsync(cv::Mat&, cv::cuda::Stream::StreamCallback, void*);
        std::vector<LatchClassifierKeypoint> identifyFeaturePoints(cv::Mat&);
        std::tuple<std::vector<LatchClassifierKeypoint>, std::vector<LatchClassifierKeypoint>, std::vector<LatchBitMatcherMatch>> identifyFeaturePointsBetweenImages(cv::Mat&, cv::Mat&);
        std::vector<LatchClassifierKeypoint> identifyFeaturePointsCPU(cv::Mat&);

        unsigned int* getDescriptorSet1() { return m_hD1; };
        unsigned int* getDescriptorSet2() { return m_hD2; };
        ~LatchClassifierCV();
	private:
        cv::cuda::Stream m_stream1;
        cv::cuda::Stream m_stream2;

};

#endif
