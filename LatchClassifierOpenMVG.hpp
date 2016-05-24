#ifndef LATCH_CLASSIFIER_OPENMVG_H
#define LATCH_CLASSIFIER_OPENMVG_H

#include <tuple>
#include <vector>

#include <Eigen/Core>

#include "LatchBitMatcher.hpp"
#include "LatchClassifier.hpp"
#include "LatchClassifierKeypoint.hpp"
#include "params.hpp"

class LatchClassifierOpenMVG : public LatchClassifier {
    public:
        LatchClassifierOpenMVG();
        // This *must* be called before identifyFeaturePoints is called
        std::vector<LatchClassifierKeypoint> identifyFeaturePointsOpenMVG(Eigen::Matrix<unsigned char, -1, -1, 1, -1, -1>);
        
        ~LatchClassifierOpenMVG();
    private:
};

#endif
