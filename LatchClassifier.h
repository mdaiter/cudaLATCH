#ifndef LATCH_CLASSIFIER_H
#define LATCH_CLASSIFIER_H

#define NUM_SM 3

#include <tuple>
#include <vector>

#include "cuda.h"
#include "cuda_runtime.h"
#include "opencv2/opencv.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/cudafeatures2d.hpp"
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace cv;

class LatchClassifier {
    public:
        LatchClassifier();
        // This *must* be called before identifyFeaturePoints is called
        void setImageSize(int, int);
        void identifyFeaturePointsAsync(Mat&, cv::cuda::Stream::StreamCallback, void*);
        std::vector<KeyPoint> identifyFeaturePoints(Mat&);
        std::tuple<std::vector<KeyPoint>, std::vector<KeyPoint>, std::vector<DMatch>> identifyFeaturePointsBetweenImages(Mat&, Mat&);
        std::vector<KeyPoint> identifyFeaturePointsCPU(Mat&);
        //std::tuple<std::vector<KeyPoint>, std::vector<KeyPoint>, std::vector<DMatch>> identifyFeaturePointsBetweenImagesCPU(Mat&, Mat&);
        void writeSIFTFile(const std::string&, int, int, unsigned int*, std::vector<cv::KeyPoint>&);
        //void writeMatFile(const string&, cv::Mat&, std::vector<cv::KeyPoint>&);
        unsigned int* getDescriptorSet1() { return m_hD1; };
        unsigned int* getDescriptorSet2() { return m_hD2; };
        ~LatchClassifier();
    private:
        // For the main portions of our class
        const int m_maxKP;
        const int m_matchThreshold;
        size_t m_pitch;

        // For the CUDA/Host specific part of our class. m_h represents host vectors. m_d represents device vectors.
        // Keypoints
        float* m_hK1;
        float* m_hK2;
        // Descriptors
        unsigned int* m_hD1;
        unsigned int* m_hD2;
        
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

        cv::Ptr<cv::ORB> m_orbClassifierCPU;
        cv::Ptr<cv::xfeatures2d::LATCH> m_latch;

        int m_width;
        int m_height;

        // For the metrics.
        bool m_shouldBeTimed;
        clock_t m_timer;
        double m_defects;
};

#endif
