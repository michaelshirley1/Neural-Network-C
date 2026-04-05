#define STB_IMAGE_IMPLEMENTATION
#include "../stb_image.h"
#include "input-helper/input.h"
#include <filesystem>
#include <stdexcept>
#include <iostream>

namespace fs = std::filesystem;

static const int TARGET_W = 32;
static const int TARGET_H = 32;

std::vector<float> input::loadImage(const std::string& path) {
    int w, h, channels;
    unsigned char* data = stbi_load(path.c_str(), &w, &h, &channels, 1);
    if (!data) {
        throw std::runtime_error("Failed to load image: " + path);
    }

    std::vector<float> result(TARGET_W * TARGET_H);
    for (int y = 0; y < TARGET_H; y++) {
        for (int x = 0; x < TARGET_W; x++) {
            int src_x = x * w / TARGET_W;
            int src_y = y * h / TARGET_H;
            result[y * TARGET_W + x] = data[src_y * w + src_x] / 255.0f;
        }
    }

    stbi_image_free(data);
    return result;
}

void input::loadDataset(const std::string& rootFolder, const std::vector<std::string>& labels, std::vector<std::vector<float>>& inputs, std::vector<std::vector<float>>& actuals) {
    for (int i = 0; i < (int)labels.size(); i++) {

        std::cout << "Hitting label " << labels[i] << std::endl;

        fs::path labelDir = fs::path(rootFolder) / labels[i];
        if (!fs::exists(labelDir) || !fs::is_directory(labelDir)) continue;

        std::vector<float> oneHot(labels.size(), 0.0f);
        oneHot[i] = 1.0f;

        for (const auto& entry : fs::directory_iterator(labelDir)) {
            std::string path = entry.path().string();
            std::string stem = entry.path().stem().string();

            try {
                if (std::stoi(stem) > 1000) continue;
            } catch (const std::invalid_argument&) {
                continue;
            } catch (const std::out_of_range&) {
                continue;
            }

            std::cout << "Hitting directory " << entry.path() << std::endl;

            std::string ext = entry.path().extension().string();
            if (ext != ".jpg" && ext != ".jpeg" && ext != ".png" && ext != ".bmp") continue;

            std::cout << "Found file " << path << std::endl;

            try {
                inputs.push_back(loadImage(entry.path().string()));
                actuals.push_back(oneHot);
            } catch (const std::exception& e) {
                std::cerr << "Failed to load image " << path << ": " << e.what() << std::endl;
            }
        }
    }
}
