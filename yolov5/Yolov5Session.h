#pragma once
#include <string>
#include <filesystem>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include "YoloDefine.h"

#include "ModelProcessor.h"
#include "ModelParser.h"
#include "ISession.h"

class Yolov5Session: public ISession
{
public:
    Yolov5Session();
    ~Yolov5Session();


    bool Initialize(const std::string& modelPath) override;

    std::vector<ResultNode> Detect(const cv::Mat& image) override;

private:
    bool CreateSession(const std::filesystem::path& modelPath);

    bool ParseModel();

    OrtCUDAProviderOptions CreateCudaOptions();

    bool IsGPUAvailable();

protected:
    bool WarmUpModel() override; // 暂时有问题

private:


    Ort::Session session_{nullptr};
    Ort::SessionOptions sessionOpt {nullptr};
    Ort::Env env_{nullptr};
    std::string envName_;

    ModelProcessor *processor_ = nullptr;
    
    bool useGpu = true;
    bool warmup = true;
};