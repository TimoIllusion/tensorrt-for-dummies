 // derived from https://github.com/NVIDIA/TensorRT/tree/release/10.6/quickstart/SemanticSegmentation

/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cassert>
#include <cfloat>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>

#include <cuda_runtime_api.h>
#include "NvInfer.h"
#include "NvOnnxParser.h"

#include "util.h"
#include "logger.h"

constexpr long long operator"" _MiB(long long unsigned val)
{
    return val * (1 << 20);
}

//!
//! \class TRTClassification
//!
//! \brief Implements semantic segmentation using FCN-ResNet101 ONNX model.
//!
class TRTClassification
{

public:
    TRTClassification(const std::string& engineFilename);
    bool infer(const std::string& input_filename, int32_t width, int32_t height, const std::string& output_filename);

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

    // De-serialize engine from file
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

//!
//! \brief Runs the TensorRT inference.
//!
//! \details Allocate input and output memory, and executes the engine.
//!
bool TRTClassification::infer(const std::string& input_filename, int32_t width, int32_t height, const std::string& output_filename)
{
    auto context = std::unique_ptr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    char const* input_name = "input";
    
    assert(mEngine->getTensorDataType(input_name) == nvinfer1::DataType::kFLOAT);
    auto input_dims = nvinfer1::Dims4{1, /* channels */ 3, height, width};
    context->setInputShape(input_name, input_dims);
    auto input_size = util::getMemorySize(input_dims, sizeof(float));

    char const* output_name = "output";
    assert(mEngine->getTensorDataType(output_name) == nvinfer1::DataType::kFLOAT);
    auto output_dims = context->getTensorShape(output_name);
    auto output_size = util::getMemorySize(output_dims, sizeof(int64_t));

    // Allocate CUDA memory for input and output bindings
    void* input_mem{nullptr};
    if (cudaMalloc(&input_mem, input_size) != cudaSuccess)
    {
        std::cout << "ERROR: input cuda memory allocation failed, size = " << input_size << " bytes" << std::endl;
        return false;
    }
    void* output_mem{nullptr};
    if (cudaMalloc(&output_mem, output_size) != cudaSuccess)
    {
        std::cout << "ERROR: output cuda memory allocation failed, size = " << output_size << " bytes" << std::endl;
        return false;
    }

    // Read image data from file and mean-normalize it
    // const std::vector<float> mean{0.485f, 0.456f, 0.406f};
    // const std::vector<float> stddev{0.229f, 0.224f, 0.225f};
    // auto input_image{util::RGBImageReader(input_filename, input_dims, mean, stddev)};
    // input_image.read();
    // auto input_buffer = input_image.process();

    // create a dummy buffer with dummy image data
    auto input_buffer = std::unique_ptr<float>{new float[input_size / sizeof(float)]};
    for (size_t i = 0; i < input_size / sizeof(float); ++i)
    {
        input_buffer.get()[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    cudaStream_t stream;
    if (cudaStreamCreate(&stream) != cudaSuccess)
    {
        std::cout << "ERROR: cuda stream creation failed." << std::endl;
        return false;
    }

    // Copy image data to input binding memory
    if (cudaMemcpyAsync(input_mem, input_buffer.get(), input_size, cudaMemcpyHostToDevice, stream) != cudaSuccess)
    {
        std::cout << "ERROR: CUDA memory copy of input failed, size = " << input_size << " bytes" << std::endl;
        return false;
    }
    context->setTensorAddress(input_name, input_mem);
    context->setTensorAddress(output_name, output_mem);

    // Run TensorRT inference
    bool status = context->enqueueV3(stream);
    if (!status)
    {
        std::cout << "ERROR: TensorRT inference failed" << std::endl;
        return false;
    }

    // Copy predictions from output binding memory
    auto output_buffer = std::unique_ptr<int64_t>{new int64_t[output_size]};
    if (cudaMemcpyAsync(output_buffer.get(), output_mem, output_size, cudaMemcpyDeviceToHost, stream) != cudaSuccess)
    {
        std::cout << "ERROR: CUDA memory copy of output failed, size = " << output_size << " bytes" << std::endl;
        return false;
    }
    cudaStreamSynchronize(stream);

    // // Plot the semantic segmentation predictions of 21 classes in a colormap image and write to file
    // const int num_classes{21};
    // const std::vector<int> palette{(0x1 << 25) - 1, (0x1 << 15) - 1, (0x1 << 21) - 1};
    // auto output_image{util::ArgmaxImageWriter(output_filename, output_dims, palette, num_classes)};
    // int64_t* output_ptr = output_buffer.get();
    // std::vector<int32_t> output_buffer_casted(output_size);
    // for (size_t i = 0; i < output_size; ++i) {
    //     output_buffer_casted[i] = static_cast<int32_t>(output_ptr[i]);
    // }
    // output_image.process(output_buffer_casted.data());
    // output_image.write();

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
    if (!sample.infer("cat.jpg", width, height, "output.ppm"))
    {
        return -1;
    }

    return 0;
}