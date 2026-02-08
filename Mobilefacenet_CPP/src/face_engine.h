#pragma once
#ifndef FACE_ENGINE_H_
#define FACE_ENGINE_H_

#include "face_detector.h"
#include "mobilefacenet.h"
#include <string>
#include <vector>
#include <map>
#include <mutex>

struct FaceRecord {
    std::string name;
    std::vector<float> feature;  // 128-dim
};

// C interface for Python ctypes
extern "C" {
    typedef void* EngineHandle;

    EngineHandle engine_create(const char* model_path, const char* db_path);
    void engine_destroy(EngineHandle h);

    // Detect faces. boxes_out: float[max_faces * 5] (x1,y1,x2,y2,score)
    int engine_detect(EngineHandle h, const unsigned char* bgr_data, int width, int height,
                      float* boxes_out, int max_faces);

    // Extract 128-d feature. feature_out: float[128]
    void engine_extract(EngineHandle h, const unsigned char* bgr_data, int width, int height,
                        float* feature_out);

    // Add face to database
    int engine_add_face(EngineHandle h, const char* name,
                        const unsigned char* bgr_data, int width, int height);

    // Verify face. Returns similarity. name_out: matched name
    float engine_verify(EngineHandle h, const unsigned char* bgr_data, int width, int height,
                        char* name_out, int name_buf_size);

    int engine_delete_face(EngineHandle h, const char* name);
    int engine_list_faces(EngineHandle h, char* names_out, int buf_size);
    int engine_save_db(EngineHandle h);
}

class FaceEngine {
public:
    FaceEngine(const std::string& model_path, const std::string& db_path);
    ~FaceEngine();

    std::vector<FaceBox> detect(const cv::Mat& img);
    std::vector<float> extract(const cv::Mat& face_crop);

    int addFace(const std::string& name, const cv::Mat& face_crop);
    std::pair<std::string, float> verify(const cv::Mat& face_crop, float threshold = 0.5f);
    int deleteFace(const std::string& name);
    std::vector<std::string> listFaces() const;
    int saveDatabase();
    int loadDatabase();

private:
    FaceDetector detector_;
    MobileFaceNet recognizer_;
    std::string db_path_;
    std::vector<FaceRecord> database_;
    mutable std::mutex db_mutex_;
};

#endif
