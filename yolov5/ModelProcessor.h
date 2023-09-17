#pragma once
#include <vector>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include "YoloDefine.h"
#include "Model.h"


class ModelProcessor
{
public:
    ModelProcessor(Model* model);
    ModelProcessor() = delete;
    ~ModelProcessor();

    std::vector<Ort::Value> Preprocess(const cv::Mat& image);

    std::vector<ResultNode> Postprocess(const std::vector<Ort::Value>& outTensor, 
            const cv::Size& originalImageShape, 
            float confThreshold, float iouThreshold);

private:
    
    cv::Mat Letterbox(const cv::Mat& image, const cv::Size& newShape = cv::Size(640, 640), 
          const cv::Scalar& color = cv::Scalar(114, 114, 114), bool scaleFill = false, bool scaleUp = true,
          int stride = 32);

    bool ConvertToRGB(const cv::Mat& image, cv::Mat& outImage);


    void GetOriCoords(const cv::Size& currentShape, 
                    const cv::Size& originalShape, cv::Rect& outCoords);

    void GetBestClassInfo(std::vector<float>::iterator it, const int& numClasses, float& bestConf, int& bestClassId);

private:

    Model* model_ = nullptr;

    float* blob_ = nullptr;
    size_t blobSize_ = 0;

    Ort::MemoryInfo memInfo_{nullptr};
};