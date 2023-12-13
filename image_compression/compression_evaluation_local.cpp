#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include "image.cpp" 

// Function to calculate file size
long getFileSize(const std::string& filename) {
    std::ifstream file(filename, std::ifstream::ate | std::ifstream::binary);
    return file.tellg(); 
}

int main() {
    std::vector<int> M_values = {4, 4, 16}; 
    std::vector<int> N_values = {4, 16, 16};  

    std::vector<std::string> inputFilenames = {
        "../images_samples/capybara.jpg",
        "../images_samples/Lenna.jpg",
        "../images_samples/beach.jpg" 
    };

    std::ofstream csvFile("compression_results_local.csv", std::ios::app); 
    if (!csvFile.is_open()) {
        std::cerr << "Unable to open file compression_results_local.csv" << std::endl;
        return 1;
    }

    csvFile.seekp(0, std::ios::end);
    if (csvFile.tellp() == 0) {
        csvFile << "InputFilename,M,N,CompressionTime,CompressionRatio\n";
    }

    for (const auto& inputFilename : inputFilenames) {
        for (size_t i = 0; i < M_values.size(); ++i) {
            int M = M_values[i];
            int N = N_values[i];

            std::string outputFilename = "output_" + inputFilename.substr(inputFilename.find_last_of("/\\") + 1) + "_M" + std::to_string(M) + "_N" + std::to_string(N) + ".jpg";

            auto start = std::chrono::high_resolution_clock::now();

            ImageProcessor processor;
            processor.processImage(inputFilename, outputFilename, M, N);

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;

            long originalSize = getFileSize(inputFilename);
            long compressedSize = getFileSize(outputFilename);
            double compressionRatio = static_cast<double>(originalSize) / compressedSize;

            csvFile << inputFilename << ","
                    << M << "," << N << "," 
                    << elapsed.count() << "," 
                    << compressionRatio << std::endl;
        }
    }

    csvFile.close();
    return 0;
}
