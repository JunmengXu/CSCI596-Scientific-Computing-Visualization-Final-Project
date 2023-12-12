#include<iostream>
#include<fstream>
#include<string>
#include <vector>
#include <random>
#include <cmath>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"

/**
 * ======= Compression using Vector Quantization =======
 * 2023 Fall, USC CSCI596 Scientific-Computing-Visualization-Final-Project
 * Team: Junmeng Xu, Yiyuan Gao
*/


int width;
int height;
int channels;
std::vector<unsigned char> img;
std::vector<unsigned char> img_compressed;


/**
 * Define a vector class for [pixel1, pixel2]
 * this is for M = 2
*/
class MyVector {
    private:
        int pixel1, pixel2;

    public:
        // Constructor
        MyVector(int pixel1, int pixel2) : pixel1(pixel1), pixel2(pixel2) {}

        // Getter for pixel1
        int getPixel1() const {
            return pixel1;
        }

        // Getter for pixel2
        int getPixel2() const {
            return pixel2;
        }

        // Calculate distance between two vectors
        double distance(const MyVector& vector) const {
            double sum = 0.0;
            double diff = pixel1 - vector.getPixel1();
            sum += diff * diff;
            diff = pixel2 - vector.getPixel2();
            sum += diff * diff;
            return std::sqrt(sum);
        }
};

/**
 * Define a vector class for M pixels
 * this is for M = perfect square
*/
class MyVectorExtra {
    private:
        std::vector<int> pixels;

    public:
        // Constructor
        MyVectorExtra(const std::vector<int>& pixels) : pixels(pixels) {}

        // Getter for pixels
        const std::vector<int>& getPixels() const {
            return pixels;
        }

        // Calculate distance between two vectors
        double distance(const MyVectorExtra& vector) const {
            double sum = 0.0;
            int M = pixels.size();
            const std::vector<int>& pixels2 = vector.getPixels();
            for (int i = 0; i < M; ++i) {
                double diff = pixels[i] - pixels2[i];
                sum += diff * diff;
            }
            return std::pow(sum, 1.0 / M);
        }
};

class ImageProcessor {
public:
    // Read the origianl file to produce the original image
    void originalImage(const std::string &filename) {

        // Load the original image using STB image
        unsigned char* rgbData = stbi_load(filename.c_str(), &width, &height, &channels, STBI_rgb);

        if (rgbData == nullptr) {
            std::cerr << "Error loading image: " << stbi_failure_reason() << std::endl;
            return;
        }

        // Initialize the global 'img' variable
        img.resize(width * height * 3, 255);


        // Set the RGB values for each pixel in the vector
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int index = (y * width + x) * 3;
                img[index] = rgbData[index];
                img[index + 1] = rgbData[index + 1];
                img[index + 2] = rgbData[index + 2];
            }
        }

        // Free allocated memory
        stbi_image_free(rgbData);
    }

    void compressImage(int M, int N, int offset) {
        // Step 1: Understanding your two-pixel vector space to see what vectors your image contains
        std::vector<MyVector> vectors;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x <= width - M; x++) {
                int pixel1 = img[(y * width + x) * 3 + offset];
                int pixel2 = img[(y * width + x + 1 ) * 3 + offset];
                MyVector vector(pixel1, pixel2);
                vectors.push_back(vector);
            }
        }
        // // Print the content of vectors
        // for (const auto& vector : vectors) {
        //     std::cout << "Pixel1: " << vector.getPixel1() << " Pixel2: " << vector.getPixel2() << std::endl;
        // }

        // Step 2: Initialization of codewords - select N initial codewords
        std::vector<MyVector> codewords;
        for (int i = 0; i < N; i++) {
            codewords.push_back(vectors[i]);
        }

        // Step 3: Clustering vectors around each code word
        // Step 4: Refine and Update your code words depending on outcome of 3
        bool change = true;
        while (change) {
            std::vector<std::vector<MyVector> > clusters(N);
            for (const auto& vector : vectors) {
                int nearestCodewordIndex = findNearestCodewordIndex(vector, codewords);
                clusters[nearestCodewordIndex].push_back(vector);
            }
            std::vector<MyVector> newCodewords;
            for (int i = 0; i < N; i++) {
                MyVector averageVector = averageVectors(clusters[i]);
                newCodewords.push_back(averageVector);
            }
            if (!codewordsChange(codewords, newCodewords, N)) {
                change = false;
                codewords = newCodewords;
            } else {
                codewords = newCodewords;
            }
        }

        // Step 5: Quantize input vectors to produce output image
        // Initialize the global 'img' variable
        img_compressed.resize(width * height * 3, 255);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x <= width - M; x++) {
                int pixel1 = img[(y * width + x) * 3 + offset];
                int pixel2 = img[(y * width + x + 1) * 3 + offset];
                MyVector vector(pixel1, pixel2);
                int nearestCodewordIndex = findNearestCodewordIndex(vector, codewords);
                MyVector quantizedVector = codewords[nearestCodewordIndex];
                int quantizedPixel1 = quantizedVector.getPixel1();
                int quantizedPixel2 = quantizedVector.getPixel2();
                img_compressed[(y * width + x) * 3 + offset] = quantizedPixel1;
                img_compressed[(y * width + x + 1) * 3 + offset] = quantizedPixel2;
            }
        }
    }

    /**
     * Compress the image by vector quantization
     * here, we assume M is perfect square, eg – M = 4 (2x2 blocks), 9 (3x3 blocks) etc.
     * and N (number of vectors) is a power of 2
    */
    void compressImageExtra(int M, int N, int offset) {
        // Step 1: Understanding your two-pixel vector space to see what vectors your image contains
		// Create list of adjacent blocks pixel vectors
        int root = static_cast<int>(std::sqrt(M));
        std::vector<MyVectorExtra> vectors;

        for (int y = 0; y <= height - root; ++y) {
            for (int x = 0; x <= width - root; ++x) {
                std::vector<int> pixels;
                for (int i = 0; i < root; ++i) {
                    for (int j = 0; j < root; ++j) {
                        pixels.push_back(img[(y * width + x + j + i * width) * 3 + offset]);
                    }
                }
                MyVectorExtra vector(pixels);
                vectors.push_back(vector);
            }
        }


        // Step 2: Initialization of codewords - select N initial codewords
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dist(0, vectors.size() - 1);

        std::vector<MyVectorExtra> codewords;
        for (int i = 0; i < N; ++i) {
            int index = dist(gen);
            codewords.push_back(vectors[index]);
        }

        // Step 3: Clustering vectors around each code word
        // Step 4: Refine and Update your code words depending on outcome of 3
        bool change = true; // whether codewords change or the change in the codewords is small.
        // Here, because of higher dimensional representations (higher M)
        // the sensitivity to change will be reduced, because too many traversals will make the program run longer
        while (change) {
            std::vector<std::vector<MyVectorExtra> > clusters(N);
            for (int i = 0; i < N; ++i) {
                clusters[i].clear();
            }

            for (const MyVectorExtra& vector : vectors) {
                int nearestCodewordIndex = findNearestCodewordIndexExtra(vector, codewords);
                clusters[nearestCodewordIndex].push_back(vector);
            }

            std::vector<MyVectorExtra> newCodewords;
            for (int i = 0; i < N; ++i) {
                MyVectorExtra averageVector = averageVectorsExtra(clusters[i], M);
                newCodewords.push_back(averageVector);
            }

            if (!codewordsChangeExtra(codewords, newCodewords, N)) {
                change = false;
                codewords = newCodewords;
            } else {
                codewords = newCodewords;
            }
        }

        // Step 5: Quantize input vectors to produce output image
        img_compressed.resize(width * height * 3, 255);

        for (int y = 0; y <= height - root; ++y) {
            for (int x = 0; x <= width - root; ++x) {
                std::vector<int> pixels;
                for (int i = 0; i < root; ++i) {
                    for (int j = 0; j < root; ++j) {
                        pixels.push_back(img[((y + i) * width + x + j) * 3 + offset]);
                    }
                }

                MyVectorExtra vector(pixels);
                int nearestCodewordIndex = findNearestCodewordIndexExtra(vector, codewords);
                MyVectorExtra quantizedVector = codewords[nearestCodewordIndex];
                std::vector<int> quantizedPixels = quantizedVector.getPixels();

                int index = 0;
                for (int i = 0; i < root; ++i) {
                    for (int j = 0; j < root; ++j) {
                        int quantizedPixel = quantizedPixels[index];
                        index++;
                        int outputIndex = ((y + i) * width + x + j) * 3 + offset;
                        img_compressed[outputIndex] = quantizedPixel;
                    }
                }
            }
        }

    }

    int findNearestCodewordIndex(const MyVector& vector, const std::vector<MyVector>& codewords) {
        int nearestIndex = 0;
        double nearestDistance = std::numeric_limits<double>::max();

        for (int i = 0; i < codewords.size(); ++i) {
            double distance = vector.distance(codewords[i]);
            if (distance < nearestDistance) {
                nearestIndex = i;
                nearestDistance = distance;
            }
        }

        return nearestIndex;
    }

    // Function to find the nearest codeword index for MyVectorExtra
    int findNearestCodewordIndexExtra(const MyVectorExtra& vector, const std::vector<MyVectorExtra>& codewords) {
        int nearestIndex = 0;
        double nearestDistance = std::numeric_limits<double>::max();

        for (int i = 0; i < codewords.size(); ++i) {
            double distance = vector.distance(codewords[i]);
            if (distance < nearestDistance) {
                nearestIndex = i;
                nearestDistance = distance;
            }
        }

        return nearestIndex;
    }

    // Updated position of each codeword by the average of each cluster
    MyVector averageVectors(const std::vector<MyVector>& list) {
        int pixel1sum = 0;
        int pixel2sum = 0;

        for (const MyVector& vector : list) {
            pixel1sum += vector.getPixel1();
            pixel2sum += vector.getPixel2();
        }

        int pixel1 = static_cast<int>(std::round(static_cast<float>(pixel1sum) / list.size()));
        int pixel2 = static_cast<int>(std::round(static_cast<float>(pixel2sum) / list.size()));

        return MyVector(pixel1, pixel2);
    }

    // Function to average vectors for MyVectorExtra
    MyVectorExtra averageVectorsExtra(const std::vector<MyVectorExtra>& list, int M) {
        std::vector<int> pixelSums(M, 0);

        for (const auto& vector : list) {
            const std::vector<int>& pixels = vector.getPixels();
            for (int j = 0; j < pixels.size(); ++j) {
                pixelSums[j] += pixels[j];
            }
        }

        std::vector<int> newPixels;
        for (int i = 0; i < M; ++i) {
            newPixels.push_back(static_cast<int>(std::round(static_cast<float>(pixelSums[i]) / list.size())));
        }

        MyVectorExtra myvector(newPixels);
        return myvector;
    }

    // Judge whether the codewords don’t change or the change in the codewords is small
    bool codewordsChange(const std::vector<MyVector>& codewords, const std::vector<MyVector>& newCodewords, int N) {
        int times = 0;

        for (int i = 0; i < codewords.size(); i++) {
            const MyVector& vectorOld = codewords[i];
            const MyVector& vectorNew = newCodewords[i];

            if (vectorOld.getPixel1() != vectorNew.getPixel1() || vectorOld.getPixel2() != vectorNew.getPixel2()) {
                times++;
            }
        }

        float timesf = static_cast<float>(times);
        float Nf = static_cast<float>(N);
        float ratio = timesf / Nf;

        if (ratio < 0.01) {
            return false;
        } else {
            return true;
        }
    }

    // Function to judge whether the codewords don’t change or the change in the codewords is small
    bool codewordsChangeExtra(const std::vector<MyVectorExtra>& codewords, const std::vector<MyVectorExtra>& newCodewords, int N) {
        int times = 0;

        for (int i = 0; i < codewords.size(); ++i) {
            const MyVectorExtra& vectorOld = codewords[i];
            const MyVectorExtra& vectorNew = newCodewords[i];

            const std::vector<int>& pixelsOld = vectorOld.getPixels();
            const std::vector<int>& pixelsNew = vectorNew.getPixels();

            bool change = false;

            for (int j = 0; j < pixelsOld.size(); ++j) {
                if (pixelsOld[j] != pixelsNew[j]) {
                    change = true;
                    break;
                }
            }

            if (change) {
                times++;
            }
        }

        float timesf = static_cast<float>(times);
        float Nf = static_cast<float>(N);
        float ratio = timesf / Nf;

        if (ratio < 0.8) {
            return false;
        } else {
            return true;
        }
    }

    // To judge whether a given number N is a power of 2 or not
    bool isPowerOf2(int N) {
        return (N & (N - 1)) == 0 && N > 0;
    }

    // To check whether a given number X is a perfect square
    bool isPerfectSquare(int X) {
        int root = static_cast<int>(std::sqrt(X));
        return root * root == X;
    }

    void outputImage(const std::string &filename) {
        // Write the compressed image to a JPEG file
        stbi_write_jpg(filename.c_str(), width, height, channels, img_compressed.data(), 100); // Assuming 3 channels (RGB) and quality 100
    }


    // Contains the whole process, read original image, compress it to make a compressed one
    // and show them both on the screen
    void processImage(const std::string& filename, const std::string& outputFilename, int M, int N) {
        originalImage(filename);

        if (!isPowerOf2(N)) {
            std::cout << "N is not a power of 2, please enter again." << std::endl;
            return;
        }
        if (M != 2 && !isPerfectSquare(M)) {
            std::cout << "M is not 2 or a perfect square, please enter again." << std::endl;
            return;
        }

        if (M == 2) {
            std::cout << "M = 2" << std::endl;
            std::cout << "Waiting for compressing..." << std::endl;
            for(int i=0; i<channels; i++){
                compressImage(M, N, i);
            }
        } else {
            std::cout << "Extra: M = perfect square" << std::endl;
            std::cout << "Waiting for compressing..." << std::endl;
            for(int i=0; i<channels; i++){
                compressImageExtra(M, N, i);
            }
        }
        std::cout << "Compressed done!" << std::endl;

        outputImage(outputFilename);
    }
};


// int main(int argc, char* argv[]) {
//     // Check if the correct number of arguments is provided
//     if (argc != 5) {
//         std::cerr << "Usage: " << argv[0] << " <input_filename> <output_filename> <M> <N>" << std::endl;
//         return 1;
//     }

//     // Extract parameters from command-line arguments
//     const std::string inputFilename = argv[1];
//     const std::string outputFilename = argv[2];
//     const int M = std::stoi(argv[3]);
//     const int N = std::stoi(argv[4]);
    
//     ImageProcessor processor;
//     processor.processImage(inputFilename, outputFilename, M, N);

//     return 0;
// }
