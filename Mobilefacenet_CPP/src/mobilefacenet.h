#pragma once
#ifndef MOBILEFACENET_H_
#define MOBILEFACENET_H_

#include <string>
#include <vector>
#include "net.h"
#include <opencv2/opencv.hpp>

class MobileFaceNet {
public:
    MobileFaceNet();
    ~MobileFaceNet();
    int load(const std::string& model_path);
    std::vector<float> extract(const cv::Mat& face_img);

private:
    ncnn::Net net_;
};

float cosine_similarity(const std::vector<float>& v1, const std::vector<float>& v2);

#endif
