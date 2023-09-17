#include "ModelProcessor.h"


ModelProcessor::ModelProcessor(Model* model)
    :model_(model)
{
    const auto& shapes = model->inputShapes;
    if(!shapes.empty())
    {
        const auto& shape = shapes[0];
        if(shape.size() == 4)
        {
            blobSize_ = shape.at(3) * shape.at(2) * shape.at(1) * shape.at(0); 
            blob_ = new float[blobSize_];
        }
    }

    memInfo_ = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
}

ModelProcessor::~ModelProcessor()
{   
    if(blob_)
        delete blob_;

    blob_  = nullptr;
}

std::vector<Ort::Value> ModelProcessor::Preprocess(const cv::Mat& image)
{
    std::vector<Ort::Value> result;
    cv::Mat resizedImage, floatImage;
    if(!model_)
        return result;
    
    auto inputShape = model_->inputShapes;
    if(inputShape.empty())
        return result;

    if(!ConvertToRGB(image, resizedImage))
        return result;

    // 只适用于yolo5、yolov8的输出结果系列
    auto inputTensorShape = inputShape[0];

    // 归一化为统一大小
    resizedImage = Letterbox(resizedImage, cv::Size(inputTensorShape.at(2), inputTensorShape.at(3)));

    // 映射 0~255到 0~1之间
    resizedImage.convertTo(floatImage, CV_32FC3, 1 / 255.0);

    cv::Size floatImageSize {floatImage.cols, floatImage.rows};

    // hwc -> chw(height width channels)
    std::vector<cv::Mat> chw(floatImage.channels());
    for (int i = 0; i < floatImage.channels(); ++i)
    {
        chw[i] = cv::Mat(floatImageSize, CV_32FC1, blob_ + i * floatImageSize.width * floatImageSize.height);
    }
    cv::split(floatImage, chw);

    //std::vector<float>inputTensorValues(blob_, blob_ + blobSize_);
    
    result.push_back(
            Ort::Value::CreateTensor<float>(memInfo_, 
            blob_, blobSize_, 
            inputTensorShape.data(), inputTensorShape.size())
        );
    
    return result;
}

std::vector<ResultNode> ModelProcessor::Postprocess(const std::vector<Ort::Value>& outTensor, 
            const cv::Size& originalImageShape, 
            float confThreshold, float iouThreshold)
{
    std::vector<cv::Rect> boxes;
    std::vector<float> confs;
    std::vector<int> classIds;
    const auto& originalShape = model_->inputShapes[0];
    cv::Size resizedImageShape = {originalShape[3], originalShape[2]};
    

    auto* rawOutput = outTensor.at(0).GetTensorData<float>();
    std::vector<int64_t> outputShape = outTensor[0].GetTensorTypeAndShapeInfo().GetShape();
    size_t count = outTensor.at(0).GetTensorTypeAndShapeInfo().GetElementCount();
    std::vector<float> output(rawOutput, rawOutput + count);

    int numClasses = (int)outputShape.at(2) - 5;
    int elementsInBatch = (int)(outputShape.at(1) * outputShape.at(2));

    // only for batch size = 1
    for (auto it = output.begin(); it != output.begin() + elementsInBatch; it += outputShape.at(2))
    {
        float clsConf = it[4];

        if (clsConf > confThreshold)
        {
            int centerX = (int) (it[0]);
            int centerY = (int) (it[1]);
            int width = (int) (it[2]);
            int height = (int) (it[3]);
            int left = centerX - width / 2;
            int top = centerY - height / 2;

            float objConf;
            int classId;
            GetBestClassInfo(it, numClasses, objConf, classId);

            float confidence = clsConf * objConf;

            boxes.emplace_back(left, top, width, height);
            confs.emplace_back(confidence);
            classIds.emplace_back(classId);
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confs, confThreshold, iouThreshold, indices);

    std::vector<ResultNode> detections;

    for (int idx : indices)
    {
        ResultNode det;
        
        GetOriCoords(resizedImageShape, originalImageShape, boxes[idx]);

        det.x = boxes[idx].x;
        det.y = boxes[idx].y;
        det.w = boxes[idx].width;
        det.h = boxes[idx].height;

        det.confidence = confs[idx];
        det.classIdx = classIds[idx];
        
        detections.emplace_back(det);
    }

    return detections;
}

cv::Mat ModelProcessor::Letterbox(const cv::Mat& image,
                      const cv::Size& newShape,
                      const cv::Scalar& color,
                      bool scaleFill,
                      bool scaleUp,
                      int stride)
{
    // 参数验证
    if (newShape.width <= 0 || newShape.height <= 0) {
        throw std::invalid_argument("Invalid newShape dimensions");
    }
    cv::Mat outImage;
    cv::Size shape = image.size();
    float scale_factor = std::min(static_cast<float>(newShape.height) / static_cast<float>(shape.height),
                              static_cast<float>(newShape.width) / static_cast<float>(shape.width));
    if (!scaleUp)
        scale_factor = std::min(scale_factor, 1.0f);

    float width_padding = (newShape.width - shape.width * scale_factor) / 2.f;
    float height_padding = (newShape.height - shape.height * scale_factor) / 2.f;


    // 如果需要填充颜色
    if (scaleFill) {
        width_padding = 0.0f;
        height_padding = 0.0f;
        scale_factor = static_cast<float>(newShape.width) / shape.width;
    }

    // 调整图像大小
    //cv::Mat outImage;
    if (shape.width != static_cast<int>(shape.width * scale_factor) && 
        shape.height != static_cast<int>(shape.height * scale_factor)) {
        cv::resize(image, outImage, cv::Size(static_cast<int>(shape.width * scale_factor),
                                             static_cast<int>(shape.height * scale_factor)));
    } else {
        outImage = image;
    }

    // 计算边界
    int top = static_cast<int>(std::round(height_padding - 0.1f));
    int bottom = static_cast<int>(std::round(height_padding + 0.1f));
    int left = static_cast<int>(std::round(width_padding - 0.1f));
    int right = static_cast<int>(std::round(width_padding + 0.1f));

    // 添加边界
    cv::copyMakeBorder(outImage, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);

    return outImage;
    /*
    float ratio[2] {scale_factor, scale_factor};
    int newUnpad[2] {(int)std::round((float)shape.width * scale_factor),
                     (int)std::round((float)shape.height * scale_factor)};

    auto dw = (float)(newShape.width - newUnpad[0]);
    auto dh = (float)(newShape.height - newUnpad[1]);


    if (scaleFill)
    {
        dw = 0.0f;
        dh = 0.0f;
        newUnpad[0] = newShape.width;
        newUnpad[1] = newShape.height;
        //ratio[0] = (float)newShape.width / (float)shape.width;
        //ratio[1] = (float)newShape.height / (float)shape.height;
    }

    dw /= 2.0f;
    dh /= 2.0f;

    if (shape.width != newUnpad[0] && shape.height != newUnpad[1])
    {
        cv::resize(image, outImage, cv::Size(newUnpad[0], newUnpad[1]));
    }

    int top = static_cast<int>(std::round(dh - 0.1f));
    int bottom = static_cast<int>(std::round(dh + 0.1f));
    int left = static_cast<int>(std::round(dw - 0.1f));
    int right = static_cast<int>(std::round(dw + 0.1f));
    cv::copyMakeBorder(outImage, outImage, top, bottom, left, right, 
        cv::BORDER_CONSTANT, color);

    return outImage;*/
}


bool ModelProcessor::ConvertToRGB(const cv::Mat& image, cv::Mat& outImage)
{
    if(image.empty())
        return false;
    
    int channel = image.channels();
    if(channel == 1)
        cv::cvtColor(image, outImage, cv::COLOR_GRAY2RGB);
    else if(channel == 3)
        cv::cvtColor(image, outImage, cv::COLOR_BGR2RGB);
    else if (channel == 4)
        cv::cvtColor(image, outImage, cv::COLOR_BGRA2RGB);
    else
        return false;
    
    return true;
}



void ModelProcessor::GetOriCoords(const cv::Size& currentShape, 
    const cv::Size& originalShape, cv::Rect& outCoords)
{
  float gain = std::min((float)currentShape.height / (float)originalShape.height,
                        (float)currentShape.width / (float)originalShape.width);

  int pad[2] = {
    (int) (((float)currentShape.width - (float)originalShape.width * gain) / 2.0f),
    (int) (((float)currentShape.height - (float)originalShape.height * gain) / 2.0f)
  };

  outCoords.x = (int) std::round(((float)(outCoords.x - pad[0]) / gain));
  outCoords.y = (int) std::round(((float)(outCoords.y - pad[1]) / gain));

  outCoords.width = (int) std::round(((float)outCoords.width / gain));
  outCoords.height = (int) std::round(((float)outCoords.height / gain));

}


void ModelProcessor::GetBestClassInfo(std::vector<float>::iterator it, 
    const int& numClasses, float& bestConf, int& bestClassId)
{
  // first 5 element are box and obj confidence
  bestClassId = 5;
  bestConf = 0;
  const int otherCnt = 5; // skip x, y, w, h, box_conf

  for (int i = otherCnt; i < numClasses + otherCnt; i++)
  {
    if (it[i] > bestConf)
    {
      bestConf = it[i];
      bestClassId = i - otherCnt;
    }
  }
}