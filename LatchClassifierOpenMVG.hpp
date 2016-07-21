#ifndef LATCH_CLASSIFIER_OPENMVG_H
#define LATCH_CLASSIFIER_OPENMVG_H

#include <tuple>
#include <vector>

#include <opencv2/core/types.hpp>
#include <Eigen/Core>

#include "LatchClassifier.hpp"
#include "LatchClassifierKeypoint.hpp"
#include "params.hpp"

class LatchClassifierOpenMVG : public LatchClassifier {
    public:
        LatchClassifierOpenMVG();
        
        std::vector<LatchClassifierKeypoint> identifyFeaturePointsOpenMVG(Eigen::Matrix<unsigned char, Eigen::Dynamic,
        Eigen::Dynamic, Eigen::RowMajor>);
        
        unsigned int* describeOpenMVG(Eigen::Matrix<unsigned char, Eigen::Dynamic,
        Eigen::Dynamic, Eigen::RowMajor>, std::vector<cv::KeyPoint>&);

        ~LatchClassifierOpenMVG();
    private:
};

#endif
