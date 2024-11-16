#include <iostream>
#include <fstream>
#include <memory>
#include <vector>
#include <algorithm>

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>

using namespace nvinfer1;

// Logger class for TensorRT info/warning/errors
class Logger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        // Only log messages with severity WARNING or higher
        if (severity >= Severity::kWARNING)
            std::cerr << msg << std::endl;
    }
};

// Function to load the TensorRT engine from file
std::unique_ptr<ICudaEngine> loadEngine(const std::string& engineFile, ILogger& logger) {
    std::ifstream file(engineFile, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening engine file: " << engineFile << std::endl;
        return nullptr;
    }
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    file.close();

    IRuntime* runtime = createInferRuntime(logger);
    ICudaEngine* engine = runtime->deserializeCudaEngine(buffer.data(), size);
    return std::unique_ptr<ICudaEngine>(engine);
}

// Function to preprocess the input image
std::vector<float> preprocessImage(const std::string& imagePath) {
    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        throw std::runtime_error("Failed to load image at " + imagePath);
    }

    // Convert BGR to RGB
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    // Resize to 224x224
    cv::resize(image, image, cv::Size(224, 224));
    // Convert to float and normalize to [0,1]
    image.convertTo(image, CV_32F, 1.0 / 255);
    // Mean and standard deviation for normalization
    cv::Scalar mean(0.485, 0.456, 0.406);
    cv::Scalar std(0.229, 0.224, 0.225);
    // Normalize the image
    cv::Mat channels[3];
    cv::split(image, channels);
    for (int i = 0; i < 3; ++i) {
        channels[i] = (channels[i] - mean[i]) / std[i];
    }
    cv::merge(channels, 3, image);
    // Convert HWC to CHW format
    std::vector<cv::Mat> chw(3);
    cv::split(image, chw);
    std::vector<float> data(3 * 224 * 224);
    for (int i = 0; i < 3; ++i) {
        std::memcpy(&data[i * 224 * 224], chw[i].data, 224 * 224 * sizeof(float));
    }
    return data;
}

int main() {
    Logger logger;

    // Load the TensorRT engine
    auto engine = loadEngine("efficientnet.engine", logger);
    if (!engine) {
        std::cerr << "Failed to load engine." << std::endl;
        return -1;
    }

    // Create execution context
    auto context = engine->createExecutionContext();
    if (!context) {
        std::cerr << "Failed to create execution context." << std::endl;
        return -1;
    }

    // Prepare input data
    std::vector<float> inputData;
    try {
        inputData = preprocessImage("input.jpg");
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }

    // Allocate device memory
    void* buffers[2];
    int inputIndex = engine->getIndex("input");
    int outputIndex = engine->getIndex("output");

    cudaMalloc(&buffers[inputIndex], inputData.size() * sizeof(float));
    cudaMalloc(&buffers[outputIndex], 1000 * sizeof(float)); // Output size for ImageNet (1000 classes)

    // Copy input data to device
    cudaMemcpy(buffers[inputIndex], inputData.data(), inputData.size() * sizeof(float), cudaMemcpyHostToDevice);

    // Execute inference
    context->enqueue(1, buffers, 0, nullptr);  // Batch size is 1

    // Copy output data back to host
    std::vector<float> outputData(1000);
    cudaMemcpy(outputData.data(), buffers[outputIndex], 1000 * sizeof(float), cudaMemcpyDeviceToHost);

    // Find the predicted class
    auto maxElementIter = std::max_element(outputData.begin(), outputData.end());
    int predictedClass = static_cast<int>(std::distance(outputData.begin(), maxElementIter));
    std::cout << "Predicted class: " << predictedClass << std::endl;

    // Clean up
    cudaFree(buffers[inputIndex]);
    cudaFree(buffers[outputIndex]);

    return 0;
}
