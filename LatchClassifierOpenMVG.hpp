#ifndef LATCH_CLASSIFIER_OPENMVG_H
#define LATCH_CLASSIFIER_OPENMVG_H

#include <tuple>
#include <vector>

#include <Eigen/Core>

#include "LatchClassifier.hpp"
#include "LatchClassifierKeypoint.hpp"
#include "params.hpp"

class LatchClassifierOpenMVG : public LatchClassifier {
    public:
        LatchClassifierOpenMVG();
        
        std::vector<LatchClassifierKeypoint> identifyFeaturePointsOpenMVG(Eigen::Matrix<unsigned char, -1, -1, 1, -1, -1>);
        
        ~LatchClassifierOpenMVG();
    private:
};

#endif
