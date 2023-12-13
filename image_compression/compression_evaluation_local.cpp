#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include "image.cpp"  // 包含您的压缩算法头文件

// Function to calculate file size
long getFileSize(const std::string& filename) {
    std::ifstream file(filename, std::ifstream::ate | std::ifstream::binary);
    return file.tellg(); 
}

int main() {
    // 定义测试的 M 和 N 值
    std::vector<int> M_values = {4, 4, 16};  // 示例 M 值
    std::vector<int> N_values = {4, 16, 16};  // 示例 N 值，确保与 M_values 长度相同

    // 定义输入文件名
    std::vector<std::string> inputFilenames = {
        "../images_samples/capybara.jpg",
        "../images_samples/Lenna.jpg",
        "../images_samples/beach.jpg"  // 输入文件名
    };

    // 打开 CSV 文件
    std::ofstream csvFile("compression_results_local.csv", std::ios::app); 
    if (!csvFile.is_open()) {
        std::cerr << "Unable to open file compression_results_local.csv" << std::endl;
        return 1;
    }

    // 检查文件是否为空，写入标题
    csvFile.seekp(0, std::ios::end);
    if (csvFile.tellp() == 0) {
        csvFile << "InputFilename,M,N,CompressionTime,CompressionRatio\n";
    }

    // 对于每个文件、每个 M 和 N 的组合
    for (const auto& inputFilename : inputFilenames) {
        for (size_t i = 0; i < M_values.size(); ++i) {
            int M = M_values[i];
            int N = N_values[i];

            // 构造输出文件名
            std::string outputFilename = "output_" + inputFilename.substr(inputFilename.find_last_of("/\\") + 1) + "_M" + std::to_string(M) + "_N" + std::to_string(N) + ".jpg";

            // 开始计时
            auto start = std::chrono::high_resolution_clock::now();

            // 处理图像
            ImageProcessor processor;
            processor.processImage(inputFilename, outputFilename, M, N);

            // 结束计时
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;

            // 计算压缩比
            long originalSize = getFileSize(inputFilename);
            long compressedSize = getFileSize(outputFilename);
            double compressionRatio = static_cast<double>(originalSize) / compressedSize;

            // 写入数据到 CSV 文件
            csvFile << inputFilename << ","
                    << M << "," << N << "," 
                    << elapsed.count() << "," 
                    << compressionRatio << std::endl;
        }
    }

    csvFile.close();
    return 0;
}
