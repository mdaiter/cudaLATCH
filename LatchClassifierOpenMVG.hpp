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
				// ONLY FOR TESTING PURPOSES FOR CVPR16 PAPER.
				int m_testingArr[48] = { 
					409,
				 	254,
					407,
					616,
					639,
					898,
					713,
					312,
					227,
					150,
					126,
					101,
					2332,
					2343,
					2018,
					1363,
					1153,
					1334,
					505,
					618,
					695,
					664,
					644,
					648,
					730,
					581,
					447,
					399,
					323,
					219,
					4780,
					5034,
					4530,
					2625,
					1268,
					696,
					1611,
					1517,
					1470,
					1439,
					1258,
					798,
					3678,
					2658,
					2443,
					2560,
					2555,
					2770
				};
			int m_count;
};

#endif
