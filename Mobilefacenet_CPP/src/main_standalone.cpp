/*
 * Pure C++ Face Recognition System for Raspberry Pi 4
 * BlazeFace (0.78MB) + MobileFaceNet (128-dim) via ncnn
 */
#include <iostream>
#include <string>
#include <chrono>
#include <deque>
#include "face_engine.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    string model_path = "../models";
    string db_path = "../database/faces.db";

    cout << "🚀 Initializing Face Engine (BlazeFace + MobileFaceNet)..." << endl;
    FaceEngine engine(model_path, db_path);

    setenv("LIBCAMERA_LOG_LEVELS", "2", 1);

    VideoCapture cap;

    // Try libcamera first, then V4L2, then default
    if (!cap.isOpened()) cap.open(0, CAP_V4L2);
    if (!cap.isOpened()) cap.open(0);
    if (!cap.isOpened()) {
        // Try /dev/video0 explicitly
        cap.open("/dev/video0", CAP_V4L2);
    }
    if (!cap.isOpened()) {
        cerr << "❌ Cannot open camera! Trying all backends..." << endl;
        for (int i = 0; i < 5; i++) {
            cap.open(i);
            if (cap.isOpened()) {
                cout << "✅ Opened camera index " << i << endl;
                break;
            }
        }
    }
    if (!cap.isOpened()) {
        cerr << "❌ No camera found at all!" << endl;
        return -1;
    }

    cap.set(CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(CAP_PROP_FPS, 30);
    cap.set(CAP_PROP_BUFFERSIZE, 2);

    int actual_w = (int)cap.get(CAP_PROP_FRAME_WIDTH);
    int actual_h = (int)cap.get(CAP_PROP_FRAME_HEIGHT);
    cout << "📷 Camera: " << actual_w << "x" << actual_h << endl;

    enum Mode { DETECT, VERIFY, ADD_WAIT };
    Mode mode = DETECT;
    string add_name;
    deque<double> fps_history;
    string last_match_name;
    float last_match_score = 0.0f;
    int match_display_frames = 0;

    cout << "\n🎮 Controls (press keys in the camera window, NOT terminal):" << endl;
    cout << "   [d] - Detection mode" << endl;
    cout << "   [v] - Verify mode" << endl;
    cout << "   [a] - Add face (then type name in terminal)" << endl;
    cout << "   [l] - List database" << endl;
    cout << "   [x] - Delete face" << endl;
    cout << "   [q] - Quit\n" << endl;

    Mat frame;

    while (true) {
        auto t0 = chrono::high_resolution_clock::now();

        cap >> frame;
        if (frame.empty()) {
            cerr << "⚠️ Empty frame, retrying..." << endl;
            continue;
        }

        // Detect faces directly on captured frame (already 640x480)
        auto faces = engine.detect(frame);

        for (auto& face : faces) {
            int dx1 = max(0, (int)face.x1);
            int dy1 = max(0, (int)face.y1);
            int dx2 = min(frame.cols, (int)face.x2);
            int dy2 = min(frame.rows, (int)face.y2);

            if (dx2 <= dx1 || dy2 <= dy1) continue;
            Mat face_crop = frame(Rect(dx1, dy1, dx2 - dx1, dy2 - dy1)).clone();

            if (mode == ADD_WAIT && !add_name.empty() && face_crop.cols > 20 && face_crop.rows > 20) {
                engine.addFace(add_name, face_crop);
                cout << "✅ Added: " << add_name << endl;
                mode = DETECT;
                add_name.clear();
            }

            Scalar color(0, 255, 0);
            string label;

            if (mode == VERIFY && face_crop.cols > 20 && face_crop.rows > 20) {
                auto result = engine.verify(face_crop);
                last_match_name = result.first;
                last_match_score = result.second;
                match_display_frames = 30;
                color = (last_match_name != "Unknown") ? Scalar(0, 255, 0) : Scalar(0, 0, 255);
                char buf[256];
                snprintf(buf, sizeof(buf), "%s (%.2f)", last_match_name.c_str(), last_match_score);
                label = buf;
            } else {
                char buf[32];
                snprintf(buf, sizeof(buf), "%.2f", face.score);
                label = buf;
            }

            rectangle(frame, Point(dx1, dy1), Point(dx2, dy2), color, 2);
            putText(frame, label, Point(dx1, dy1 - 10), FONT_HERSHEY_SIMPLEX, 0.6, color, 2);

            // Draw landmarks
            for (int k = 0; k < 5; ++k) {
                circle(frame, Point((int)face.landmark[k].x, (int)face.landmark[k].y),
                       2, Scalar(0, 255, 255), -1);
            }
        }

        // FPS
        auto t1 = chrono::high_resolution_clock::now();
        double elapsed = chrono::duration<double>(t1 - t0).count();
        fps_history.push_back(1.0 / max(elapsed, 0.001));
        if (fps_history.size() > 30) fps_history.pop_front();
        double avg_fps = 0;
        for (auto f : fps_history) avg_fps += f;
        avg_fps /= fps_history.size();

        char hud[128];
        snprintf(hud, sizeof(hud), "FPS: %.1f", avg_fps);
        putText(frame, hud, Point(10, 25), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 255), 2);

        const char* mode_str = (mode == DETECT) ? "DETECT" : (mode == VERIFY) ? "VERIFY" : "ADD";
        snprintf(hud, sizeof(hud), "Mode: %s | Faces: %d", mode_str, (int)faces.size());
        putText(frame, hud, Point(10, 50), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 0), 2);

        if (match_display_frames > 0) {
            snprintf(hud, sizeof(hud), "Match: %s (%.3f)", last_match_name.c_str(), last_match_score);
            Scalar c = (last_match_name != "Unknown") ? Scalar(0, 255, 0) : Scalar(0, 0, 255);
            putText(frame, hud, Point(10, 75), FONT_HERSHEY_SIMPLEX, 0.6, c, 2);
            match_display_frames--;
        }

        if (mode == ADD_WAIT) {
            snprintf(hud, sizeof(hud), "Adding: %s - look at camera!", add_name.c_str());
            putText(frame, hud, Point(10, frame.rows - 15), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 165, 255), 2);
        }

        imshow("Face Recognition - RPi4", frame);

        // waitKey must be called for imshow to work — keys are read HERE not from terminal
        int key = waitKey(1) & 0xFF;
        if (key == 'q') break;
        else if (key == 'd') { mode = DETECT; cout << "🔍 Detection mode" << endl; }
        else if (key == 'v') { mode = VERIFY; cout << "🔐 Verify mode" << endl; }
        else if (key == 'a') {
            // Show a prompt frame so user knows to look at terminal
            putText(frame, ">>> Type name in TERMINAL then press Enter <<<",
                    Point(10, frame.rows / 2), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255), 2);
            imshow("Face Recognition - RPi4", frame);
            waitKey(100);
            cout << "Enter name: " << flush;
            cin >> add_name;
            if (!add_name.empty()) {
                mode = ADD_WAIT;
                cout << "📸 Look at camera to add '" << add_name << "'..." << endl;
            }
        }
        else if (key == 'l') {
            auto names = engine.listFaces();
            cout << "📋 Database (" << names.size() << " faces):" << endl;
            for (auto& n : names) cout << "   - " << n << endl;
        }
        else if (key == 'x') {
            cout << "Delete name: " << flush;
            string del;
            cin >> del;
            cout << (engine.deleteFace(del) == 0 ? "✅ Deleted" : "❌ Not found") << endl;
        }
    }

    cap.release();
    destroyAllWindows();
    cout << "👋 Bye!" << endl;
    return 0;
}
