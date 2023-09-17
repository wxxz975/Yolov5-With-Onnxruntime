#pragma once
#include <vector>
#include <string>

#include <opencv2/opencv.hpp>

#include "YoloDefine.h"
#include "Model.h"

class ISession
{
public:
    ISession() = default;
    ~ISession() = default;


public:
    virtual bool Initialize(const std::string& modelPath) = 0;

    virtual std::vector<ResultNode> Detect(const cv::Mat& image) = 0;

    virtual Model* GetModel()  { return model_; };

protected:
    Model *model_;
};