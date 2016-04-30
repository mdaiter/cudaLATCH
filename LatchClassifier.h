#ifndef LATCH_CLASSIFIER_H
#define LATCH_CLASSIFIER_H

#define NUM_SM 15

#include "latch.h"

class LatchClassifier {
    public:
        LatchClassifier();
        identifyFeaturePoints();
        ~LatchClassifier();
    private:
        // For the main portions of our class
        const int m_maxKP = 512 * NUM_SM;
        const int m_bitMapThreshold = 12;
        

        /* For the FAST detector. In the future, the detector input should be able to be changed to a general OpenCV
         abstract classifier class. */
        int m_detectorThreshold;
        int m_targetKP;
        int m_tolerance;

        // For the metrics. Currently only adding in defects
        double defect;
};

#endif
