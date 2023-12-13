#include <iostream>
#include <fstream>
#include <chrono>
#include "image_omp.cpp"  // Assuming your compression algorithm is in this header file

// Function to calculate file size
long getFileSize(const std::string& filename) {
    std::ifstream file(filename, std::ifstream::ate | std::ifstream::binary);
    return file.tellg(); 
}

int main(int argc, char* argv[]) {
    if (argc != 8) {
        std::cerr << "Usage: " << argv[0] << " <input_filename> <output_filename> <M> <N> <nodes> <ntasks> <cpus_per_task>" << std::endl;
        return 1;
    }

    const std::string inputFilename = argv[1];
    const std::string outputFilename = argv[2];
    const int M = std::stoi(argv[3]);
    const int N = std::stoi(argv[4]);

    const int nodes = std::stoi(argv[5]);
    const int ntasks = std::stoi(argv[6]);
    const int cpus_per_task = std::stoi(argv[7]);

    ImageProcessor processor;

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    processor.processImage(inputFilename, outputFilename, M, N);

    // End timing
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Calculate compression ratio and PSNR
    long originalSize = getFileSize(inputFilename);
    long compressedSize = getFileSize(outputFilename);
    double compressionRatio = static_cast<double>(originalSize) / compressedSize;

    // Open or create CSV file
    std::ofstream csvFile("compression_results.csv", std::ios::app); // Append mode
    if (!csvFile.is_open()) {
        std::cerr << "Unable to open file compression_results.csv" << std::endl;
        return 1;
    }

    // Check if the file is empty to write headers
    csvFile.seekp(0, std::ios::end); // Go to the end of the file
    if (csvFile.tellp() == 0) { // File is empty
        // Write headers
        csvFile << "InputFilename,M,N,CompressionTime,CompressionRatio,Nodes,Ntasks,CpusPerTask\n";
    }

    // Write data to CSV file
    csvFile << inputFilename << ","
            << M << "," << N << "," 
            << elapsed.count() << "," 
            << compressionRatio << "," 
            << nodes << "," << ntasks << "," << cpus_per_task << std::endl;

    csvFile.close();
    return 0;
}
