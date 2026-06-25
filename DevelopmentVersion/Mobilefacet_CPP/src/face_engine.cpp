#include "face_engine.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstring>
#include <algorithm>

FaceEngine::FaceEngine(const std::string& model_path, const std::string& db_path)
    : db_path_(db_path)
{
    int ret = detector_.load(model_path);
    if (ret != 0) {
        std::cerr << "❌ Failed to load face detector: " << ret << std::endl;
    } else {
        std::cout << "✅ Face detector loaded" << std::endl;
    }

    ret = recognizer_.load(model_path);
    if (ret != 0) {
        std::cerr << "❌ Failed to load face recognizer: " << ret << std::endl;
    } else {
        std::cout << "✅ Face recognizer loaded" << std::endl;
    }

    loadDatabase();
}

FaceEngine::~FaceEngine() {
    saveDatabase();
}

std::vector<FaceBox> FaceEngine::detect(const cv::Mat& img) {
    return detector_.detect(img);
}

std::vector<float> FaceEngine::extract(const cv::Mat& face_crop) {
    return recognizer_.extract(face_crop);
}

int FaceEngine::addFace(const std::string& name, const cv::Mat& face_crop) {
    std::lock_guard<std::mutex> lock(db_mutex_);

    // Check if name already exists, update if so
    auto it = std::find_if(database_.begin(), database_.end(),
        [&name](const FaceRecord& r) { return r.name == name; });

    std::vector<float> feature = recognizer_.extract(face_crop);
    if (feature.empty()) return -1;

    if (it != database_.end()) {
        it->feature = feature;
        std::cout << "📝 Updated face: " << name << std::endl;
    } else {
        database_.push_back({name, feature});
        std::cout << "➕ Added face: " << name << std::endl;
    }

    saveDatabase();
    return 0;
}

std::pair<std::string, float> FaceEngine::verify(const cv::Mat& face_crop, float threshold) {
    std::lock_guard<std::mutex> lock(db_mutex_);

    std::vector<float> feature = recognizer_.extract(face_crop);
    if (feature.empty()) return {"", -1.0f};

    std::string best_name = "Unknown";
    float best_score = -1.0f;

    for (const auto& record : database_) {
        float sim = cosine_similarity(feature, record.feature);
        if (sim > best_score) {
            best_score = sim;
            best_name = record.name;
        }
    }

    if (best_score < threshold) {
        best_name = "Unknown";
    }

    return {best_name, best_score};
}

int FaceEngine::deleteFace(const std::string& name) {
    std::lock_guard<std::mutex> lock(db_mutex_);

    auto it = std::remove_if(database_.begin(), database_.end(),
        [&name](const FaceRecord& r) { return r.name == name; });

    if (it == database_.end()) return -1;

    database_.erase(it, database_.end());
    saveDatabase();
    std::cout << "🗑️ Deleted face: " << name << std::endl;
    return 0;
}

std::vector<std::string> FaceEngine::listFaces() const {
    std::lock_guard<std::mutex> lock(db_mutex_);
    std::vector<std::string> names;
    for (const auto& r : database_) {
        names.push_back(r.name);
    }
    return names;
}

int FaceEngine::saveDatabase() {
    std::ofstream ofs(db_path_, std::ios::binary);
    if (!ofs.is_open()) return -1;

    int count = database_.size();
    ofs.write((char*)&count, sizeof(int));

    for (const auto& record : database_) {
        int name_len = record.name.size();
        ofs.write((char*)&name_len, sizeof(int));
        ofs.write(record.name.c_str(), name_len);
        ofs.write((char*)record.feature.data(), 128 * sizeof(float));
    }

    ofs.close();
    return 0;
}

int FaceEngine::loadDatabase() {
    std::ifstream ifs(db_path_, std::ios::binary);
    if (!ifs.is_open()) {
        std::cout << "ℹ️ No existing database found, starting fresh." << std::endl;
        return 0;
    }

    int count = 0;
    ifs.read((char*)&count, sizeof(int));

    database_.clear();
    for (int i = 0; i < count; i++) {
        FaceRecord record;
        int name_len = 0;
        ifs.read((char*)&name_len, sizeof(int));
        record.name.resize(name_len);
        ifs.read(&record.name[0], name_len);
        record.feature.resize(128);
        ifs.read((char*)record.feature.data(), 128 * sizeof(float));
        database_.push_back(record);
    }

    std::cout << "📂 Loaded " << count << " faces from database." << std::endl;
    ifs.close();
    return 0;
}

// ============ C API Implementation ============

extern "C" {

EngineHandle engine_create(const char* model_path, const char* db_path) {
    return new FaceEngine(model_path, db_path);
}

void engine_destroy(EngineHandle h) {
    delete static_cast<FaceEngine*>(h);
}

int engine_detect(EngineHandle h, const unsigned char* bgr_data, int width, int height,
                  float* boxes_out, int max_faces) {
    FaceEngine* engine = static_cast<FaceEngine*>(h);
    cv::Mat img(height, width, CV_8UC3, const_cast<unsigned char*>(bgr_data));
    auto faces = engine->detect(img);

    int n = std::min((int)faces.size(), max_faces);
    for (int i = 0; i < n; i++) {
        boxes_out[i * 5 + 0] = faces[i].x1;
        boxes_out[i * 5 + 1] = faces[i].y1;
        boxes_out[i * 5 + 2] = faces[i].x2;
        boxes_out[i * 5 + 3] = faces[i].y2;
        boxes_out[i * 5 + 4] = faces[i].score;
    }
    return n;
}

void engine_extract(EngineHandle h, const unsigned char* bgr_data, int width, int height,
                    float* feature_out) {
    FaceEngine* engine = static_cast<FaceEngine*>(h);
    cv::Mat img(height, width, CV_8UC3, const_cast<unsigned char*>(bgr_data));
    auto feature = engine->extract(img);
    memcpy(feature_out, feature.data(), 128 * sizeof(float));
}

int engine_add_face(EngineHandle h, const char* name,
                    const unsigned char* bgr_data, int width, int height) {
    FaceEngine* engine = static_cast<FaceEngine*>(h);
    cv::Mat img(height, width, CV_8UC3, const_cast<unsigned char*>(bgr_data));
    return engine->addFace(name, img);
}

float engine_verify(EngineHandle h, const unsigned char* bgr_data, int width, int height,
                    char* name_out, int name_buf_size) {
    FaceEngine* engine = static_cast<FaceEngine*>(h);
    cv::Mat img(height, width, CV_8UC3, const_cast<unsigned char*>(bgr_data));
    auto result = engine->verify(img);
    strncpy(name_out, result.first.c_str(), name_buf_size - 1);
    name_out[name_buf_size - 1] = '\0';
    return result.second;
}

int engine_delete_face(EngineHandle h, const char* name) {
    FaceEngine* engine = static_cast<FaceEngine*>(h);
    return engine->deleteFace(name);
}

int engine_list_faces(EngineHandle h, char* names_out, int buf_size) {
    FaceEngine* engine = static_cast<FaceEngine*>(h);
    auto names = engine->listFaces();
    std::string joined;
    for (const auto& n : names) {
        if (!joined.empty()) joined += "\n";
        joined += n;
    }
    strncpy(names_out, joined.c_str(), buf_size - 1);
    names_out[buf_size - 1] = '\0';
    return names.size();
}

int engine_save_db(EngineHandle h) {
    FaceEngine* engine = static_cast<FaceEngine*>(h);
    return engine->saveDatabase();
}

} // extern "C"
