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
    bool infer(const std::string& input_filename, int32_t width, int32_t height, int iterations = 100);

private:
    std::string mEngineFilename;                    
    nvinfer1::Dims mInputDims;                      
    nvinfer1::Dims mOutputDims;                     
    std::unique_ptr<TRTLogger> mLogger;             
    std::unique_ptr<nvinfer1::IRuntime> mRuntime;  
    std::unique_ptr<nvinfer1::ICudaEngine> mEngine;
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
}

bool TRTInference::infer(const std::string& input_filename, int32_t width, int32_t height, int iterations)
{
    auto context = std::unique_ptr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    char const* input_name = "input";
    assert(mEngine->getTensorDataType(input_name) == nvinfer1::DataType::kFLOAT);
    auto input_dims = nvinfer1::Dims4{1, 3, height, width};
    context->setInputShape(input_name, input_dims);
    auto input_size = util::getMemorySize(input_dims, sizeof(float));

    char const* output_name = "output";
    assert(mEngine->getTensorDataType(output_name) == nvinfer1::DataType::kFLOAT);
    auto output_dims = context->getTensorShape(output_name);
    auto output_size = util::getMemorySize(output_dims, sizeof(float));

    void* input_mem{nullptr};
    cudaMalloc(&input_mem, input_size);
    void* output_mem{nullptr};
    cudaMalloc(&output_mem, output_size);

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

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaMemcpyAsync(input_mem, input_buffer.data(), input_size, cudaMemcpyHostToDevice, stream);

    context->setTensorAddress(input_name, input_mem);
    context->setTensorAddress(output_name, output_mem);

    std::vector<double> times;
    std::cout << "Running TensorRT inference..." << std::endl;

    for (int i = 0; i < iterations; ++i)
    {
        auto start_time = std::chrono::high_resolution_clock::now();
        if (!context->enqueueV3(stream))
        {
            std::cout << "ERROR: TensorRT inference failed" << std::endl;
            return false;
        }
        cudaStreamSynchronize(stream);
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> inference_time = end_time - start_time;
        times.push_back(inference_time.count());
        std::cout << "Iteration " << i + 1 << ": " << inference_time.count() << "ms" << std::endl;
    }

    double avg_time = std::accumulate(times.begin() + 5, times.end(), 0.0) / (iterations - 5);
    std::cout << "\nAverage inference time (last " << iterations - 5 << " runs): " << avg_time << "ms" << std::endl;

    std::vector<float> output_buffer(output_size / sizeof(float));
    cudaMemcpyAsync(output_buffer.data(), output_mem, output_size, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    postprocessOutput(output_buffer);

    cudaFree(input_mem);
    cudaFree(output_mem);
    return true;
}

int main(int argc, char** argv)
{
    int32_t width{224};
    int32_t height{224};

    TRTInference sample("efficientnet.engine");

    std::cout << "Running TensorRT inference" << std::endl;
    if (!sample.infer("cat.jpg", width, height, 100))
    {
        return -1;
    }

    return 0;
}
