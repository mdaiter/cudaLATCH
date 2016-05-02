#include <vector>
#include <iostream>
#include <time.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "opencv2/opencv.hpp"
using namespace std;
using namespace cv;
#include "latch.h"
#include "bitMatcher.h"
#include "LatchClassifier.h"

#define cudaCalloc(A, B) \
    do { \
        cudaError_t __cudaCalloc_err = cudaMalloc(A, B); \
        if (__cudaCalloc_err == cudaSuccess) cudaMemset(*A, 0, B); \
    } while (0)

std::string outputFilenameJPGString(int i, int j, int k) {
    std::string filenameBase = "cam(";
    filenameBase += std::to_string(i);
    filenameBase += ",";
    filenameBase += std::to_string(j);
    filenameBase += ")_";
    filenameBase += std::to_string(k);
    filenameBase += ".jpg";
    return filenameBase;
}

int main( int argc, char** argv ) {
    // maKP is the maximum number of keypoints/features you will be able to use on the GPU.
    // This _must_ be an integer multiple of 512.
    // Integers which are themselves a multiple of the number of streaming multiprocessors
    // on your GPU (or half that number) should work well.
    clock_t t;

    Mat img1, img2, imgMatches;
 
    LatchClassifier latchClass;

    img1 = imread(outputFilenameJPGString(1, 1, 0), IMREAD_COLOR);

    const int imgWidth = img1.cols; // Assumes both images are the same size.
    const int imgHeight = img1.rows;
    
    // We know all images will be the same size
    latchClass.setImageSize(imgWidth, imgHeight);


    // Loop over all images in directory. Create comparisons based on looping
    for (size_t i = 1; i < 9; i++) {
        for (size_t j = 1; j < 9; j++) {
            for (size_t k = 0; k < 18; k++) {
                std::string filename = outputFilenameJPGString(i, j, k);
                img1 = imread(outputFilenameJPGString(i, j, k), IMREAD_COLOR);
                // And now we start doing the main main loop
                for (size_t a = i; a < 9; a++) {
                    for (size_t b = j; b < 9; b++) {
                        for (size_t c = k; c < 19; c++) {
                            img2 = imread(outputFilenameJPGString(a, b, c), IMREAD_COLOR);

                            t = clock(); // Begin timing kernel launches.
                            
                            // Put as much CPU code as possible here.
                            // The CPU can continue to do useful work while the GPU is thinking.
                            // If you put no code here, the CPU will stall until the GPU is done.

                            auto keypointVectors = latchClass.identifyFeaturePointsBetweenImages(img1, img2);
                            cout << "Gathering results took " << 1000*(clock() - t)/(float)CLOCKS_PER_SEC << " milliseconds." << endl;


                        }
                    }
                }
            }
        }
    }


    waitKey(0);

    return 0;
}
