#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <memory>
#include <string>
#include <vector>
#include <chrono>
#include <numeric>

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity != Severity::kINFO)
            std::cout << msg << std::endl;
    }
} gLogger;

class TensorRTInference {
private:
    static constexpr int INPUT_H = 224;
    static constexpr int INPUT_W = 224;
    static constexpr int OUTPUT_SIZE = 1000; // ImageNet classes

    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    cudaStream_t stream;
    
    void* buffers[2];
    int inputSize;
    int outputSize;

public:
    TensorRTInference(const std::string& enginePath) {
        // Load engine file
        std::ifstream file(enginePath, std::ios::binary);
        if (!file.good()) {
            throw std::runtime_error("Engine file not found: " + enginePath);
        }

        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);
        std::vector<char> engineData(size);
        file.read(engineData.data(), size);

        // Create runtime and engine
        runtime.reset(nvinfer1::createInferRuntime(gLogger));
        engine.reset(runtime->deserializeCudaEngine(engineData.data(), size));
        context.reset(engine->createExecutionContext());

        // Create CUDA stream
        cudaStreamCreate(&stream);

        // Allocate buffers
        inputSize = 3 * INPUT_H * INPUT_W * sizeof(float);
        outputSize = OUTPUT_SIZE * sizeof(float);
        cudaMalloc(&buffers[0], inputSize);
        cudaMalloc(&buffers[1], outputSize);
    }

    ~TensorRTInference() {
        cudaFree(buffers[0]);
        cudaFree(buffers[1]);
        cudaStreamDestroy(stream);
    }

    cv::Mat preprocess(const cv::Mat& image) {
        cv::Mat resized, float_img;
        cv::resize(image, resized, cv::Size(INPUT_W, INPUT_H));
        resized.convertTo(float_img, CV_32F, 1.0/255.0);
        
        // Normalize using ImageNet stats
        std::vector<float> mean = {0.485f, 0.456f, 0.406f};
        std::vector<float> std = {0.229f, 0.224f, 0.225f};
        
        cv::Mat channels[3];
        cv::split(float_img, channels);
        for (int i = 0; i < 3; i++) {
            channels[i] = (channels[i] - mean[i]) / std[i];
        }
        cv::merge(channels, 3, float_img);
        return float_img;
    }

    int inference(const cv::Mat& preprocessed) {
        // Copy input to GPU
        cudaMemcpyAsync(buffers[0], preprocessed.data, inputSize, 
                       cudaMemcpyHostToDevice, stream);

        // Run inference
        context->enqueueV2(buffers, stream, nullptr);

        // Copy output from GPU
        std::vector<float> output(OUTPUT_SIZE);
        cudaMemcpyAsync(output.data(), buffers[1], outputSize, 
                       cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        // Get max class
        return std::max_element(output.begin(), output.end()) - output.begin();
    }
};

int main() {
    try {
        TensorRTInference trt("efficientnet.engine");
        cv::Mat image = cv::imread("goldfish.jpg");
        if (image.empty()) {
            throw std::runtime_error("Failed to load image");
        }

        // Preprocess image
        cv::Mat preprocessed = trt.preprocess(image);
        
        std::vector<double> times;
        std::cout << "Running inference 10 times...\n";

        // Run inference loop
        for (int i = 0; i < 10; i++) {
            auto start = std::chrono::high_resolution_clock::now();
            
            int class_id = trt.inference(preprocessed);
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>
                          (end - start).count();
            
            times.push_back(duration);
            std::cout << "Iteration " << i+1 << ": " << duration << "ms\n";
        }

        // Calculate average of last 5 runs
        double avg = std::accumulate(times.end()-5, times.end(), 0.0) / 5.0;
        std::cout << "\nAverage inference time (last 5 runs): " << avg << "ms\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}