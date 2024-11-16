 // derived from https://github.com/NVIDIA/TensorRT/tree/release/10.6/quickstart/SemanticSegmentation (Apache-2.0)

#include <cassert>
#include <cfloat>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>
#include <algorithm>

#include <cuda_runtime_api.h>
#include "NvInfer.h"
#include "NvOnnxParser.h"

#include "util.h"
#include "logger.h"
#include <opencv2/opencv.hpp>

constexpr long long operator"" _MiB(long long unsigned val)
{
    return val * (1 << 20);
}

//!
//! \class TRTClassification
//!
//! \brief Implements classification using TensorRT.
//!
class TRTClassification
{
public:
    TRTClassification(const std::string& engineFilename);
    bool infer(const std::string& input_filename, int32_t width, int32_t height);

private:
    std::string mEngineFilename;                    //!< Filename of the serialized engine.

    nvinfer1::Dims mInputDims;                      //!< The dimensions of the input to the network.
    nvinfer1::Dims mOutputDims;                     //!< The dimensions of the output to the network.

    std::unique_ptr<TRTLogger> mLogger;             //!< The TensorRT logger used during engine build.
    std::unique_ptr<nvinfer1::IRuntime> mRuntime;   //!< The TensorRT runtime used to run the network
    std::unique_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network
};

TRTClassification::TRTClassification(const std::string& engineFilename)
    : mEngineFilename(engineFilename)
    , mEngine(nullptr), mLogger(new TRTLogger(nvinfer1::ILogger::Severity::kINFO))
{
    // Deserialize engine from file
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

// Function to preprocess the input image
std::vector<float> preprocessImage(const std::string& imagePath, int32_t width, int32_t height)
{
    cv::Mat image = cv::imread(imagePath);
    if (image.empty())
    {
        throw std::runtime_error("Failed to load image at " + imagePath);
    }

    // Convert BGR to RGB
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    // Resize image
    cv::resize(image, image, cv::Size(width, height));
    // Normalize image
    image.convertTo(image, CV_32F, 1.0 / 255);
    cv::Scalar mean(0.485, 0.456, 0.406);
    cv::Scalar stddev(0.229, 0.224, 0.225);
    cv::Mat channels[3];
    cv::split(image, channels);
    for (int i = 0; i < 3; ++i)
    {
        channels[i] = (channels[i] - mean[i]) / stddev[i];
    }
    cv::merge(channels, 3, image);

    // Convert HWC to CHW format
    std::vector<cv::Mat> chw(3);
    cv::split(image, chw);
    std::vector<float> data(3 * width * height);
    for (int i = 0; i < 3; ++i)
    {
        std::memcpy(&data[i * width * height], chw[i].data, width * height * sizeof(float));
    }

    return data;
}

//!
//! \brief Runs the TensorRT inference.
//!
bool TRTClassification::infer(const std::string& input_filename, int32_t width, int32_t height)
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

    // Allocate CUDA memory for input and output bindings
    void* input_mem{nullptr};
    cudaMalloc(&input_mem, input_size);
    void* output_mem{nullptr};
    cudaMalloc(&output_mem, output_size);

    // Preprocess image
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

    // Copy image data to input binding memory
    cudaMemcpyAsync(input_mem, input_buffer.data(), input_size, cudaMemcpyHostToDevice, stream);

    context->setTensorAddress(input_name, input_mem);
    context->setTensorAddress(output_name, output_mem);

    // Run TensorRT inference
    std::cout << "Enqueueing job" << std::endl;
    if (!context->enqueueV3(stream))
    {
        std::cout << "ERROR: TensorRT inference failed" << std::endl;
        return false;
    }

    // Copy predictions from output binding memory
    std::vector<float> output_buffer(output_size / sizeof(float));
    cudaMemcpyAsync(output_buffer.data(), output_mem, output_size, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // Postprocess output: Find the top-1 class
    auto max_iter = std::max_element(output_buffer.begin(), output_buffer.end());
    int top1_class = std::distance(output_buffer.begin(), max_iter);
    float confidence = *max_iter;

    std::cout << "Predicted class: " << top1_class << " with confidence: " << confidence << std::endl;

    // Free CUDA resources
    cudaFree(input_mem);
    cudaFree(output_mem);
    return true;
}

int main(int argc, char** argv)
{
    int32_t width{224};
    int32_t height{224};

    TRTClassification sample("efficientnet.engine");

    std::cout << "Running TensorRT inference" << std::endl;
    if (!sample.infer("cat.jpg", width, height))
    {
        return -1;
    }

    return 0;
}
