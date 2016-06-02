#ifndef LATCH_H
#define LATCH_H

#include <vector>
#include "cuda.h"
#include "cuda_runtime.h"

void latchGPU(cv::cuda::GpuMat,
            unsigned char*,
            size_t,
            float *,
            unsigned int *,
            int *,
            int,
            float *,
            std::vector<cv::KeyPoint>*,
            float*,
            cudaStream_t,
            cudaEvent_t);


void latch( cv::Mat,
            unsigned char *,
            size_t,
            float *,
            unsigned int *,
            int *,
            int,
            float *,
            std::vector<cv::KeyPoint>*,
            float*,
            cudaEvent_t);

void loadPatchTriplets(cudaArray*);

void initImage(    unsigned char**,
                    int,
                    int,
                    size_t *
                );

void initMask(      float **,
                    float *);

#endif
