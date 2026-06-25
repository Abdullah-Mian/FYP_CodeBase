#pragma once
#ifndef FACE_DETECTOR_H_
#define FACE_DETECTOR_H_

#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include "net.h"
#include <opencv2/opencv.hpp>

struct FacePoint {
    float x;
    float y;
};

struct FaceBox {
    float x1, y1, x2, y2;
    float score;
    FacePoint landmark[5];  // 5-point face landmarks
};

// Anchor box for BlazeFace
struct AnchorBox {
    float cx, cy, sx, sy;
};

class FaceDetector {
public:
    FaceDetector();
    ~FaceDetector();
    int load(const std::string& model_path);
    std::vector<FaceBox> detect(const cv::Mat& bgr_img, float score_threshold = 0.6f, float nms_threshold = 0.4f);

private:
    void createAnchors(std::vector<AnchorBox>& anchors, int w, int h);
    void nms(std::vector<FaceBox>& boxes, float threshold);
    static bool cmpScore(const FaceBox& a, const FaceBox& b);

    ncnn::Net* net_;
    int target_size_;
    float mean_vals_[3];
};

#endif
