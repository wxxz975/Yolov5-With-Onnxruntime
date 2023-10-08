#pragma once
#include <vector>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include "YoloDefine.h"
#include "Model.h"


class ModelProcessor
{
public:
    ModelProcessor(Model* model);
    ModelProcessor() = delete;
    ~ModelProcessor();

    // yolov5 模型的预处理函数(归一化图像存储格式为RGB、图像大小为模型指定大小、生成Onnxruntime需要的Ort::Value类型并且返回)
    std::vector<Ort::Value> Preprocess(const cv::Mat& image);

    // yolov5后处理(主要是读取原始onnxruntime生成的数据并解析后经nms处理 的到符合阈值的结果集合并返回)
    std::vector<ResultNode> Postprocess(const std::vector<Ort::Value>& outTensor, 
            const cv::Size& originalImageShape, 
            float confThreshold, float iouThreshold);

private:
    
    // 将图像归一画为统一大小，主要是符合这个模型的输入维度的尺寸
    cv::Mat Letterbox(const cv::Mat& image, const cv::Size& newShape = cv::Size(640, 640), 
          const cv::Scalar& color = cv::Scalar(114, 114, 114), bool scaleFill = false, bool scaleUp = true,
          int stride = 32);

    // 图像由opencv读取的格式， 转换为RGB
    bool ConvertToRGB(const cv::Mat& image, cv::Mat& outImage);


    // 计算原始图像中的坐标
    void GetOriCoords(const cv::Size& currentShape, 
                    const cv::Size& originalShape, cv::Rect& outCoords);

    // 获取 当前框中 score 最高的一个
    void GetBestClassInfo(std::vector<float>::iterator it, const int& numClasses, float& bestConf, int& bestClassId);


    void ParseRawOutput(const std::vector<Ort::Value>& tensor, float conf_threshold,std::vector<cv::Rect>& boxes, std::vector<float>& confs, std::vector<int>& classIds);

private:

    Model* model_ = nullptr;

    float* blob_ = nullptr;
    size_t blobSize_ = 0;

    Ort::MemoryInfo memInfo_{nullptr};
};