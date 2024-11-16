#ifndef PREPOSTPROCESSING_H
#define PREPOSTPROCESSING_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <iostream>

// Preprocessing function to handle image loading, resizing, and normalization
std::vector<float> preprocessImage(const std::string& imagePath, int32_t width, int32_t height)
{
    cv::Mat image = cv::imread(imagePath);
    if (image.empty())
    {
        throw std::runtime_error("Failed to load image at " + imagePath);
    }

    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    cv::resize(image, image, cv::Size(width, height));
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

    std::vector<cv::Mat> chw(3);
    cv::split(image, chw);
    std::vector<float> data(3 * width * height);
    for (int i = 0; i < 3; ++i)
    {
        std::memcpy(&data[i * width * height], chw[i].data, width * height * sizeof(float));
    }

    return data;
}

// Postprocessing function to handle output interpretation
void postprocessOutput(const std::vector<float>& outputBuffer)
{
    auto maxIter = std::max_element(outputBuffer.begin(), outputBuffer.end());
    int topClass = std::distance(outputBuffer.begin(), maxIter);
    float confidence = *maxIter;

    std::cout << "Predicted class: " << topClass << " with confidence: " << confidence << std::endl;
}

#endif // PREPOSTPROCESSING_H