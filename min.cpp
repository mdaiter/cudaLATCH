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
    std::string filenameBase = "./cam(";
    filenameBase += std::to_string(i);
    filenameBase += ",";
    filenameBase += std::to_string(j);
    filenameBase += ")_";
    filenameBase += std::to_string(k);
    filenameBase += ".jpg";
    return filenameBase;
}

void writeNewMatchToFile(FILE*& fileHandle, 
                         std::string filename1,
                         std::string filename2,
                         std::vector<cv::DMatch> features) {
    fprintf(fileHandle, "%s %s %d\n", filename1.c_str(), filename2.c_str(), features.size());
    for (size_t i = 0; i < features.size(); i++) {
        cv::DMatch currentFeature = features.at(i);
        fprintf(fileHandle, "%d ", currentFeature.queryIdx);
    }
    fprintf(fileHandle, "\n");
    for (size_t i = 0; i < features.size(); i++) {
        cv::DMatch currentFeature = features.at(i);
        fprintf(fileHandle, "%d ", currentFeature.trainIdx);
    }
    fprintf(fileHandle, "\n");
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
    FILE* matchesFile = fopen("matches.txt", "wb");

    Mat img1, img2, imgMatches;
 
    LatchClassifier latchClass;

    // We know all images will be the same size
    latchClass.setImageSize(4000, 3000);

    // Loop over all images in directory. Create comparisons based on looping
    for (size_t i = index; i < 2; i++) {
        for (size_t j = 1; j < 2; j++) {
            for (size_t k = 0; k < 4; k++) {
                std::string filename = outputFilenameJPGString(i, j, k);
                img1 = imread(filename, IMREAD_COLOR);
                auto keypoints = latchClass.identifyFeaturePoints(img1);
                latchClass.writeSIFTFile(filename.substr(0, filename.length() - 4) + ".sift", img1.cols, img1.rows, latchClass.getDescriptorSet1(), keypoints);
                // And now we start doing the main main loop
                for (size_t a = i; a < 2; a++) {
                    for (size_t b = j; b < 2; b++) {
                        for (size_t c = k + 1; c < 4; c++) {
                            std::string filename2 = outputFilenameJPGString(a, b, c);
                            img2 = imread(filename2, IMREAD_COLOR);

                            t = clock(); // Begin timing kernel launches.
                            
                            // Put as much CPU code as possible here.
                            // The CPU can continue to do useful work while the GPU is thinking.
                            // If you put no code here, the CPU will stall until the GPU is done.

                            auto keypointsVector = latchClass.identifyFeaturePointsBetweenImages(img1, img2);
                            writeNewMatchToFile(matchesFile, filename, filename2, std::get<2>(keypointsVector));
                            cout << "Time taken: " << 1000*(clock() - t)/(float)CLOCKS_PER_SEC
                                 << " with size: " << std::get<2>(keypointsVector).size() <<  endl;
                        }
                    }
                }
            }
        }
    }
    fclose(matchesFile);
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
