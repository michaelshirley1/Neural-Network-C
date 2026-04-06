#pragma once
#include <vector>
#include <string>
#include <map>

class input {
public:
    static std::vector<float> loadImage(const std::string& path);
    void static loadDataset(const std::string& rootFolder, const std::vector<std::string>& labels, std::vector<std::vector<float>>& inputs, std::vector<std::vector<float>>& actuals);
};