#ifndef LATCH_CLASSIFIER_H
#define LATCH_CLASSIFIER_H

#define NUM_SM 5

#include <tuple>
#include <vector>

#include "cuda.h"
#include "cuda_runtime.h"
#include "opencv2/opencv.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/cudafeatures2d.hpp"

using namespace cv;

class LatchClassifier {
    public:
        LatchClassifier();
        // This *must* be called before identifyFeaturePoints is called
        void setImageSize(int, int);
        void identifyFeaturePointsAsync(Mat&, cv::cuda::Stream::StreamCallback, void*);
        std::vector<KeyPoint> identifyFeaturePoints(Mat&);
        std::tuple<std::vector<KeyPoint>, std::vector<KeyPoint>, std::vector<DMatch>> identifyFeaturePointsBetweenImages(Mat&, Mat&);
        ~LatchClassifier();
    private:
        // For the main portions of our class
        const int m_maxKP;
        const int m_matchThreshold;
        size_t m_pitch;

        // For the CUDA/Host specific part of our class. m_h represents host vectors. m_d represents device vectors.
        float* m_hK1;
        float* m_hK2;
        
        unsigned char* m_dI;
        unsigned int* m_dD1;
        unsigned int* m_dD2;
        unsigned int* m_dUIntSwapPointer; // Only necessary as an aid when swapping m_dD1 and m_dD2.
        int* m_dM1;
        int* m_dM2;
        float* m_dK;
        float* m_dMask;
        
        cudaArray* m_patchTriplets;
        cudaEvent_t m_latchFinished;
        // Used for two image comparison
        cv::cuda::Stream& m_stream1;
        cv::cuda::Stream& m_stream2;
        // Used for one image comparison
        cv::cuda::Stream& m_stream;

        /* For the FAST/ORB detector. In the future, the detector input should be able to be changed to a general OpenCV
         abstract classifier class (or at least GPU::ORB). */
        cv::Ptr<cv::cuda::ORB> m_orbClassifier;
        int m_detectorThreshold;
        int m_detectorTargetKP;
        int m_detectorTolerance;

        int m_width;
        int m_height;

        // For the metrics.
        bool m_shouldBeTimed;
        clock_t m_timer;
        double m_defects;
};

#endif
