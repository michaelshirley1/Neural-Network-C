#pragma once
#include <vector>
#include <string>

class input {
public:
    std::vector<float> loadImage(const std::string& path);
    void loadDataset(const std::string& rootFolder, const std::vector<std::string>& labels, std::vector<std::vector<float>>& inputs, std::vector<std::vector<float>>& actuals);
};