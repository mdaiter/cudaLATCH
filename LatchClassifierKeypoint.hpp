#ifndef LATCH_CLASSIFIER_KEYPOINT_H
#define LATCH_CLASSIFIER_KEYPOINT_H

struct LatchClassifierKeypoint {
    LatchClassifierKeypoint() {
       x = 0.0f;
       y = 0.0f;
       angle = 0.0f;
       size = 0.0f;
    }
    LatchClassifierKeypoint(float _x, float _y, float _angle, float _size) 
        : x(_x),
          y(_y),
          angle(_angle),
          size(_size) {}
    float x, y;
    float angle;
    float size;
};

#endif
