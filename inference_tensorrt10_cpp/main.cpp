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

#include "util.h"
#include "logger.h"
#include "prepostprocessing.h" // Header file for preprocessing and postprocessing functions

class TRTInference
{
public:
    TRTInference(const std::string& engineFilename);
    bool infer(const std::string& input_filename, int32_t width, int32_t height);

private:
    std::string mEngineFilename;                    
    nvinfer1::Dims mInputDims;                      
    nvinfer1::Dims mOutputDims;                     
    std::unique_ptr<TRTLogger> mLogger;             
    std::unique_ptr<nvinfer1::IRuntime> mRuntime;  
    std::unique_ptr<nvinfer1::ICudaEngine> mEngine;
    std::unique_ptr<nvinfer1::IExecutionContext> mContext; // Execution context for inference
    void* mInputMem{nullptr}; // CUDA memory for input
    void* mOutputMem{nullptr}; // CUDA memory for output
    size_t mInputSize; // Size of input memory
    size_t mOutputSize; // Size of output memory
    cudaStream_t mStream; // CUDA stream

    void allocateResources(int32_t width, int32_t height);
};

TRTInference::TRTInference(const std::string& engineFilename)
    : mEngineFilename(engineFilename)
    , mEngine(nullptr), mLogger(new TRTLogger(nvinfer1::ILogger::Severity::kINFO))
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

    mRuntime.reset(nvinfer1::createInferRuntime(*mLogger));
    mEngine.reset(mRuntime->deserializeCudaEngine(engineData.data(), fsize));
    assert(mEngine.get() != nullptr);

    mContext.reset(mEngine->createExecutionContext());
    assert(mContext.get() != nullptr);

    // Allocate resources once during initialization
    allocateResources(224, 224); // Assuming default width and height, can be adjusted as needed
    cudaStreamCreate(&mStream);
}

void TRTInference::allocateResources(int32_t width, int32_t height)
{
    char const* input_name = "input";
    assert(mEngine->getTensorDataType(input_name) == nvinfer1::DataType::kFLOAT);
    auto input_dims = nvinfer1::Dims4{1, 3, height, width};
    mContext->setInputShape(input_name, input_dims);
    mInputSize = util::getMemorySize(input_dims, sizeof(float));

    char const* output_name = "output";
    assert(mEngine->getTensorDataType(output_name) == nvinfer1::DataType::kFLOAT);
    auto output_dims = mContext->getTensorShape(output_name);
    mOutputSize = util::getMemorySize(output_dims, sizeof(float));

    cudaMalloc(&mInputMem, mInputSize);
    cudaMalloc(&mOutputMem, mOutputSize);
}

bool TRTInference::infer(const std::string& input_filename, int32_t width, int32_t height)
{
    std::vector<float> input_buffer;
    try
    {
        input_buffer = preprocessImage(input_filename, width, height);
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return false;
    }

    cudaMemcpyAsync(mInputMem, input_buffer.data(), mInputSize, cudaMemcpyHostToDevice, mStream);

    mContext->setTensorAddress("input", mInputMem);
    mContext->setTensorAddress("output", mOutputMem);

    std::cout << "Running TensorRT inference..." << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();
    if (!mContext->enqueueV3(mStream))
    {
        std::cout << "ERROR: TensorRT inference failed" << std::endl;
        return false;
    }
    cudaStreamSynchronize(mStream);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> inference_time = end_time - start_time;
    std::cout << "Inference time: " << inference_time.count() << "ms" << std::endl;

    std::vector<float> output_buffer(mOutputSize / sizeof(float));
    cudaMemcpyAsync(output_buffer.data(), mOutputMem, mOutputSize, cudaMemcpyDeviceToHost, mStream);
    cudaStreamSynchronize(mStream);

    postprocessOutput(output_buffer);

    return true;
}

int main(int argc, char** argv)
{
    int32_t width{224};
    int32_t height{224};

    TRTInference sample("efficientnet.engine");

    std::cout << "Running TensorRT inference" << std::endl;

    int iterations = 100;
    std::vector<double> times;
    for (int i = 0; i < iterations; ++i)
    {
        auto start_time = std::chrono::high_resolution_clock::now();
        if (!sample.infer("cat.jpg", width, height))
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

    return 0;
}
