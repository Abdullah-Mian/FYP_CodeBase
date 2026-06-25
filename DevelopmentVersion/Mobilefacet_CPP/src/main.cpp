#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "net.h"

using namespace std;
using namespace cv;

int main() {
    // Load NCNN model
    ncnn::Net net;
    
    net.opt.num_threads = 4;
    net.opt.use_vulkan_compute = false;
    
    int ret_param = net.load_param("../models/mobilefacenet.param");
    int ret_bin = net.load_model("../models/mobilefacenet.bin");
    
    if (ret_param != 0 || ret_bin != 0) {
        cerr << "Failed to load model!" << endl;
        return -1;
    }
    
    cout << "✅ MobileFaceNet loaded successfully!" << endl;
    
    // Test with dummy input (112x112 RGB)
    ncnn::Mat in = ncnn::Mat(112, 112, 3);
    in.fill(0.5f);
    
    // Create extractor
    ncnn::Extractor ex = net.create_extractor();
    ex.input("data", in);
    
    // Extract output
    ncnn::Mat out;
    ex.extract("fc1", out);
    
    cout << "✅ Inference successful!" << endl;
    cout << "   Output dimension: " << out.w << endl;
    
    return 0;
}
