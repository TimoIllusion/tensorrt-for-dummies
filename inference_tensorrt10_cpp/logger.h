#ifndef TRT_LOGGER_H
#define TRT_LOGGER_H

#include <NvInfer.h>
#include <iostream>
#include <string>

// TensorRT Logger Class
class TRTLogger : public nvinfer1::ILogger {
public:
    TRTLogger(nvinfer1::ILogger::Severity severity = nvinfer1::ILogger::Severity::kWARNING)
        : reportableSeverity(severity) {}

    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override {
        // Filter out messages with severity lower than the set severity
        if (severity > reportableSeverity) return;

        // Print severity level
        std::string severityString;
        switch (severity) {
            case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
                severityString = "INTERNAL_ERROR";
                break;
            case nvinfer1::ILogger::Severity::kERROR:
                severityString = "ERROR";
                break;
            case nvinfer1::ILogger::Severity::kWARNING:
                severityString = "WARNING";
                break;
            case nvinfer1::ILogger::Severity::kINFO:
                severityString = "INFO";
                break;
            case nvinfer1::ILogger::Severity::kVERBOSE:
                severityString = "VERBOSE";
                break;
            default:
                severityString = "UNKNOWN";
                break;
        }

        // Print the log message
        std::cerr << "[TensorRT " << severityString << "] " << msg << std::endl;
    }

private:
    nvinfer1::ILogger::Severity reportableSeverity;
};

#endif // TRT_LOGGER_H
