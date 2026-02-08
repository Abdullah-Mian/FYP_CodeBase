#include "mobilefacenet.h"
#include <cmath>
#include <cassert>

MobileFaceNet::MobileFaceNet() {}

MobileFaceNet::~MobileFaceNet() {
    net_.clear();
}

int MobileFaceNet::load(const std::string& model_path) {
    std::string param = model_path + "/mobilefacenet.param";
    std::string bin = model_path + "/mobilefacenet.bin";

    int ret = net_.load_param(param.c_str());
    if (ret != 0) return -1;
    ret = net_.load_model(bin.c_str());
    if (ret != 0) return -2;

    net_.opt.num_threads = 4;
    net_.opt.use_vulkan_compute = false;

    return 0;
}

std::vector<float> MobileFaceNet::extract(const cv::Mat& face_img) {
    // Resize to 112x112 and convert BGR -> RGB
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(
        face_img.data, ncnn::Mat::PIXEL_BGR2RGB,
        face_img.cols, face_img.rows, 112, 112
    );

    // MobileFaceNet from MXNet: already has mean subtraction in the network
    // If your model is from caffe, uncomment these:
    // const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    // const float norm_vals[3] = {1.0f/128.0f, 1.0f/128.0f, 1.0f/128.0f};
    // in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = net_.create_extractor();
    ex.set_light_mode(true);
    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("fc1", out);

    std::vector<float> feature(128);
    for (int i = 0; i < 128; i++) {
        feature[i] = out[i];
    }

    // L2 normalize
    float norm = 0.0f;
    for (float f : feature) norm += f * f;
    norm = std::sqrt(norm);
    if (norm > 0) {
        for (float& f : feature) f /= norm;
    }

    return feature;
}

float cosine_similarity(const std::vector<float>& v1, const std::vector<float>& v2) {
    assert(v1.size() == v2.size());
    float dot = 0.0f, mod1 = 0.0f, mod2 = 0.0f;
    for (size_t i = 0; i < v1.size(); i++) {
        dot += v1[i] * v2[i];
        mod1 += v1[i] * v1[i];
        mod2 += v2[i] * v2[i];
    }
    return dot / (std::sqrt(mod1) * std::sqrt(mod2));
}
