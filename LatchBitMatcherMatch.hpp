#ifndef LATCH_BIT_MATCHER_MATCH
#define LATCH_BIT_MATCHER_MATCH

struct LatchBitMatcherMatch {
    LatchBitMatcherMatch() :
        queryIdx(0),
        trainIdx(0),
        distance(0.0) {
    }

    LatchBitMatcherMatch(int _queryIdx, int _trainIdx, int _distance) :
        queryIdx(_queryIdx),
        trainIdx(_trainIdx),
        distance(_distance) {

    }

    int queryIdx;
    int trainIdx;
    float distance;
};

#endif
