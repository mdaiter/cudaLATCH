#include <pthread.h>
#include <vector>
#include <iostream>
#include <time.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "opencv2/xfeatures2d/nonfree.hpp"
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

void* compare(void* data) {
     // maKP is the maximum number of keypoints/features you will be able to use on the GPU.
    // This _must_ be an integer multiple of 512.
    // Integers which are themselves a multiple of the number of streaming multiprocessors
    // on your GPU (or half that number) should work well.
    int* indexPtr = (int*) data;
    const int index = *indexPtr;
    clock_t total_time_elapsed = clock();
    clock_t t;

    Mat img1, img2, imgMatches;
 
    LatchClassifier latchClass;

    // We know all images will be the same size
    latchClass.setImageSize(4000, 3000);

    // Loop over all images in directory. Create comparisons based on looping
    for (size_t i = index; i < 10; i++) {
        for (size_t j = 1; j < 10; j++) {
            for (size_t k = 0; k < 19; k++) {
                std::string filename = outputFilenameJPGString(i, j, k);
                img1 = imread(filename, IMREAD_COLOR);
                // And now we start doing the main main loop
                for (size_t a = i; a < 10; a++) {
                    for (size_t b = j; b < 10; b++) {
                        for (size_t c = k; c < 19; c++) {
                            img2 = imread(outputFilenameJPGString(a, b, c), IMREAD_COLOR);

                            t = clock(); // Begin timing kernel launches.
                            
                            // Put as much CPU code as possible here.
                            // The CPU can continue to do useful work while the GPU is thinking.
                            // If you put no code here, the CPU will stall until the GPU is done.

                            auto keypointsVector = latchClass.identifyFeaturePointsBetweenImages(img1, img2);
                            unsigned int* descriptorsForOne = latchClass.getDescriptorSet1();
                            latchClass.writeSIFTFile(filename + ".sift", img1, std::get<0>(keypointsVector));
                            cout << "Time taken: " << 1000*(clock() - t)/(float)CLOCKS_PER_SEC
                                 << " with size: " << std::get<2>(keypointsVector).size() <<  endl;
                        }
                    }
                }
            }
        }
    }
    cout << "Total time elapsed: " << 1000*(clock() - total_time_elapsed)/(float)CLOCKS_PER_SEC << endl;
	waitKey(0);
}

int main( int argc, char** argv ) {
    const int num_threads = 1;

    pthread_t threads[num_threads];

    for (int i = 0; i < num_threads; i++) {
        int t = i + 1;
        if (pthread_create(&threads[i], NULL, compare, &t)) {
            fprintf(stderr, "Error creating threadn");
            return 1;
        }
    }

    for (int i = 0; i < num_threads; i++) {
        if(pthread_join(threads[i], NULL)) {
            fprintf(stderr, "Error joining threadn");
            return 2;
        }
    }

    cudaDeviceReset();

    return 0;
}
