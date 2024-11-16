// derived from https://github.com/NVIDIA/TensorRT/tree/release/10.6/quickstart/SemanticSegmentation (Apache-2.0)

#include <cassert>
#include <cfloat>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <numeric>

#include <cuda_runtime_api.h>
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include <opencv2/opencv.hpp>

#include "prepostprocessing.h" // Header file for preprocessing and postprocessing functions

// Simple TensorRT Logger Class
class SimpleLogger : public nvinfer1::ILogger {
public:
    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override {
        std::cerr << "[TensorRT] " << msg << std::endl;
    }
};

class TRTInference
{
public:
    TRTInference(const std::string& engineFilename, nvinfer1::ILogger& logger);
    bool infer(const std::vector<float>& input_buffer, std::vector<float>& output_buffer);

private:
    std::string mEngineFilename;                    
    nvinfer1::Dims mInputDims;                      
    nvinfer1::Dims mOutputDims;                     
    nvinfer1::ILogger& mLogger;              
    std::unique_ptr<nvinfer1::IRuntime> mRuntime;  
    std::unique_ptr<nvinfer1::ICudaEngine> mEngine;
    std::unique_ptr<nvinfer1::IExecutionContext> mContext; // Execution context for inference
    void* mInputMem{nullptr}; // Pinned host memory for input
    void* mOutputMem{nullptr}; // Device memory for output
    size_t mInputSize; // Size of input memory
    size_t mOutputSize; // Size of output memory
    cudaStream_t mStream; // CUDA stream

    void allocateResources(int32_t width, int32_t height);
    static size_t getMemorySize(const nvinfer1::Dims& dims, const int32_t elem_size);
};

TRTInference::TRTInference(const std::string& engineFilename, nvinfer1::ILogger& logger)
    : mEngineFilename(engineFilename)
    , mEngine(nullptr), mLogger(logger)
{
    std::ifstream engineFile(engineFilename, std::ios::binary);
    if (engineFile.fail())
    {
        return;
    }

    engineFile.seekg(0, std::ifstream::end);
    auto fsize = engineFile.tellg();
    engineFile.seekg(0, std::ifstream::beg);

    std::vector<char> engineData(fsize);
    engineFile.read(engineData.data(), fsize);

    mRuntime.reset(nvinfer1::createInferRuntime(mLogger));
    mEngine.reset(mRuntime->deserializeCudaEngine(engineData.data(), fsize));
    assert(mEngine.get() != nullptr);

    mContext.reset(mEngine->createExecutionContext());
    assert(mContext.get() != nullptr);

    // Allocate resources once during initialization
    allocateResources(224, 224); // Assuming default width and height, can be adjusted as needed
    cudaStreamCreateWithPriority(&mStream, cudaStreamNonBlocking, 1); // Use high-priority non-blocking stream
}

void TRTInference::allocateResources(int32_t width, int32_t height)
{
    char const* input_name = "input";
    assert(mEngine->getTensorDataType(input_name) == nvinfer1::DataType::kFLOAT);
    auto input_dims = nvinfer1::Dims4{1, 3, height, width};
    mContext->setInputShape(input_name, input_dims);
    mInputSize = getMemorySize(input_dims, sizeof(float));

    char const* output_name = "output";
    assert(mEngine->getTensorDataType(output_name) == nvinfer1::DataType::kFLOAT);
    auto output_dims = mContext->getTensorShape(output_name);
    mOutputSize = getMemorySize(output_dims, sizeof(float));

    // Allocate pinned memory for input and device memory for output
    cudaHostAlloc(&mInputMem, mInputSize, cudaHostAllocDefault);  // Allocate pinned host memory for input
    cudaMalloc(&mOutputMem, mOutputSize);  // Device memory for output
}

bool TRTInference::infer(const std::vector<float>& input_buffer, std::vector<float>& output_buffer)
{
    cudaMemcpyAsync(mInputMem, input_buffer.data(), mInputSize, cudaMemcpyHostToDevice, mStream);

    mContext->setTensorAddress("input", mInputMem);
    mContext->setTensorAddress("output", mOutputMem);

    std::cout << "Running TensorRT inference..." << std::endl;

    // Launch inference asynchronously
    if (!mContext->enqueueV3(mStream))
    {
        std::cout << "ERROR: TensorRT inference failed" << std::endl;
        return false;
    }

    // Asynchronous copy output back to host while inference is running
    output_buffer.resize(mOutputSize / sizeof(float));
    cudaMemcpyAsync(output_buffer.data(), mOutputMem, mOutputSize, cudaMemcpyDeviceToHost, mStream);

    // Synchronize once at the end
    cudaStreamSynchronize(mStream);

    return true;
}

size_t TRTInference::getMemorySize(const nvinfer1::Dims& dims, const int32_t elem_size)
{
    return std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int64_t>()) * elem_size;
}

int main(int argc, char** argv)
{
    int32_t width{224};
    int32_t height{224};

    SimpleLogger logger;
    TRTInference sample("efficientnet.engine", logger);

    std::cout << "Running TensorRT inference" << std::endl;

    // Preprocess the image once
    std::vector<float> input_buffer;
    try
    {
        input_buffer = preprocessImage("cat.jpg", width, height);
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return -1;
    }

    int iterations = 100;
    std::vector<double> times;
    std::vector<float> output_buffer;
    for (int i = 0; i < iterations; ++i)
    {
        auto start_time = std::chrono::high_resolution_clock::now();
        if (!sample.infer(input_buffer, output_buffer))
        {
            return -1;
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> inference_time = end_time - start_time;
        times.push_back(inference_time.count());
        std::cout << "Iteration " << i + 1 << ": " << inference_time.count() << "ms" << std::endl;
    }

    double avg_time = std::accumulate(times.begin() + 5, times.end(), 0.0) / (iterations - 5);
    std::cout << "\nAverage inference time (last " << iterations - 5 << " runs): " << avg_time << "ms" << std::endl;

    // Postprocess the output after inference
    postprocessOutput(output_buffer);

    return 0;
}
