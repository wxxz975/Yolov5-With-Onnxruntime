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
    // 初始化 推理会话 、 解析模型
    virtual bool Initialize(const std::string& modelPath) = 0;

    // 推理
    virtual std::vector<ResultNode> Detect(const cv::Mat& image) = 0;

    // 获取模型参数
    virtual Model* GetModel()  { return model_; };

    // 设置confidence 阈值
    virtual void SetConfidence(float conf) { confidenceThreshold_ = conf; };

    // 设置iou阈值
    virtual void SetIOU(float iou) { iouThreshold_ = iou; };

protected:
    virtual bool WarmUpModel() = 0;

protected:
    Model *model_;

    float confidenceThreshold_;
    float iouThreshold_;
};