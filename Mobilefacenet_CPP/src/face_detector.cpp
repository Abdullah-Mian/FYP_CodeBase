// BlazeFace detector for ncnn
// Adapted from https://github.com/zineos/blazeface/blob/main/demo_ncnn/FaceDetector.cpp

#include "face_detector.h"

FaceDetector::FaceDetector()
    : net_(nullptr), target_size_(320)
{
    mean_vals_[0] = 104.f;
    mean_vals_[1] = 117.f;
    mean_vals_[2] = 123.f;
}

FaceDetector::~FaceDetector() {
    if (net_) {
        net_->clear();
        delete net_;
        net_ = nullptr;
    }
}

int FaceDetector::load(const std::string& model_path) {
    net_ = new ncnn::Net();
    net_->opt.num_threads = 4;
    net_->opt.use_vulkan_compute = false;

    std::string param = model_path + "/blazeface.param";
    std::string bin = model_path + "/blazeface.bin";

    int ret = net_->load_param(param.c_str());
    if (ret != 0) {
        delete net_;
        net_ = nullptr;
        return -1;
    }
    ret = net_->load_model(bin.c_str());
    if (ret != 0) {
        delete net_;
        net_ = nullptr;
        return -2;
    }
    return 0;
}

bool FaceDetector::cmpScore(const FaceBox& a, const FaceBox& b) {
    return a.score > b.score;
}

void FaceDetector::createAnchors(std::vector<AnchorBox>& anchors, int w, int h) {
    anchors.clear();
    // BlazeFace anchor config: 2 feature maps, steps 8 and 16
    float steps[] = {8, 16};
    std::vector<std::vector<int>> min_sizes(2);
    min_sizes[0] = {8, 11};
    min_sizes[1] = {14, 19, 26, 38, 64, 149};

    for (int k = 0; k < 2; ++k) {
        int feat_h = (int)ceil((float)h / steps[k]);
        int feat_w = (int)ceil((float)w / steps[k]);
        for (int i = 0; i < feat_h; ++i) {
            for (int j = 0; j < feat_w; ++j) {
                for (int l = 0; l < (int)min_sizes[k].size(); ++l) {
                    float s_kx = (float)min_sizes[k][l] / w;
                    float s_ky = (float)min_sizes[k][l] / h;
                    float cx = (j + 0.5f) * steps[k] / w;
                    float cy = (i + 0.5f) * steps[k] / h;
                    anchors.push_back({cx, cy, s_kx, s_ky});
                }
            }
        }
    }
}

void FaceDetector::nms(std::vector<FaceBox>& boxes, float threshold) {
    std::vector<float> areas(boxes.size());
    for (size_t i = 0; i < boxes.size(); ++i) {
        areas[i] = (boxes[i].x2 - boxes[i].x1 + 1) * (boxes[i].y2 - boxes[i].y1 + 1);
    }
    for (size_t i = 0; i < boxes.size(); ++i) {
        for (size_t j = i + 1; j < boxes.size();) {
            float xx1 = std::max(boxes[i].x1, boxes[j].x1);
            float yy1 = std::max(boxes[i].y1, boxes[j].y1);
            float xx2 = std::min(boxes[i].x2, boxes[j].x2);
            float yy2 = std::min(boxes[i].y2, boxes[j].y2);
            float w = std::max(0.0f, xx2 - xx1 + 1);
            float h = std::max(0.0f, yy2 - yy1 + 1);
            float inter = w * h;
            float ovr = inter / (areas[i] + areas[j] - inter);
            if (ovr >= threshold) {
                boxes.erase(boxes.begin() + j);
                areas.erase(areas.begin() + j);
            } else {
                j++;
            }
        }
    }
}

std::vector<FaceBox> FaceDetector::detect(const cv::Mat& bgr_img, float score_threshold, float nms_threshold) {
    std::vector<FaceBox> results;
    if (!net_) return results;

    int img_w = bgr_img.cols;
    int img_h = bgr_img.rows;

    // Letterbox resize: scale longest edge to target_size_
    int w = img_w, h = img_h;
    float scale = 1.0f;
    if (w > h) {
        scale = (float)target_size_ / w;
        w = target_size_;
        h = (int)(h * scale);
    } else {
        scale = (float)target_size_ / h;
        h = target_size_;
        w = (int)(w * scale);
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr_img.data, ncnn::Mat::PIXEL_BGR,
                                                  img_w, img_h, w, h);

    // Pad to multiple of 16
    int wpad = (w + 15) / 16 * 16 - w;
    int hpad = (h + 15) / 16 * 16 - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2,
                           wpad / 2, wpad - wpad / 2,
                           ncnn::BORDER_CONSTANT, 114.f);

    in_pad.substract_mean_normalize(mean_vals_, 0);

    // Run inference
    ncnn::Extractor ex = net_->create_extractor();
    ex.set_light_mode(true);

    ex.input(0, in_pad);

    ncnn::Mat out_boxes, out_scores, out_landmarks;
    ex.extract("boxes", out_boxes);
    ex.extract("scores", out_scores);
    ex.extract("landmark", out_landmarks);

    // Generate anchors
    std::vector<AnchorBox> anchors;
    createAnchors(anchors, w, h);

    // Decode detections
    float* ptr_box = out_boxes.channel(0);
    float* ptr_score = out_scores.channel(0);
    float* ptr_lm = out_landmarks.channel(0);

    std::vector<FaceBox> candidates;

    for (size_t i = 0; i < anchors.size(); ++i) {
        float face_score = *(ptr_score + 1);
        if (face_score > score_threshold) {
            AnchorBox& a = anchors[i];
            FaceBox fb;

            // Decode box
            float cx = a.cx + (*ptr_box) * 0.1f * a.sx;
            float cy = a.cy + (*(ptr_box + 1)) * 0.1f * a.sy;
            float sw = a.sx * exp((*(ptr_box + 2)) * 0.2f);
            float sh = a.sy * exp((*(ptr_box + 3)) * 0.2f);

            fb.x1 = (cx - sw / 2.0f) * in.w;
            fb.y1 = (cy - sh / 2.0f) * in.h;
            fb.x2 = (cx + sw / 2.0f) * in.w;
            fb.y2 = (cy + sh / 2.0f) * in.h;
            fb.score = face_score;

            // Clamp
            fb.x1 = std::max(0.0f, fb.x1);
            fb.y1 = std::max(0.0f, fb.y1);
            fb.x2 = std::min((float)in.w, fb.x2);
            fb.y2 = std::min((float)in.h, fb.y2);

            // Decode 5 landmarks
            for (int j = 0; j < 5; ++j) {
                fb.landmark[j].x = (a.cx + (*(ptr_lm + j * 2)) * 0.1f * a.sx) * in.w;
                fb.landmark[j].y = (a.cy + (*(ptr_lm + j * 2 + 1)) * 0.1f * a.sy) * in.h;
            }

            candidates.push_back(fb);
        }
        ptr_box += 4;
        ptr_score += 2;
        ptr_lm += 10;
    }

    // Sort by score and NMS
    std::sort(candidates.begin(), candidates.end(), cmpScore);
    nms(candidates, nms_threshold);

    // Unpad and unscale back to original image coordinates
    for (auto& fb : candidates) {
        fb.x1 = (fb.x1 - wpad / 2.0f) / scale;
        fb.y1 = (fb.y1 - hpad / 2.0f) / scale;
        fb.x2 = (fb.x2 - wpad / 2.0f) / scale;
        fb.y2 = (fb.y2 - hpad / 2.0f) / scale;

        for (int k = 0; k < 5; ++k) {
            fb.landmark[k].x = (fb.landmark[k].x - wpad / 2.0f) / scale;
            fb.landmark[k].y = (fb.landmark[k].y - hpad / 2.0f) / scale;
        }

        // Clamp to image bounds
        fb.x1 = std::max(0.0f, std::min(fb.x1, (float)img_w));
        fb.y1 = std::max(0.0f, std::min(fb.y1, (float)img_h));
        fb.x2 = std::max(0.0f, std::min(fb.x2, (float)img_w));
        fb.y2 = std::max(0.0f, std::min(fb.y2, (float)img_h));

        results.push_back(fb);
    }

    return results;
}
