

#include "yolov5/Yolov5Session.h"

#include <iostream>

int main(int argc, char* argv[])
{
    ISession *session = new Yolov5Session();
    bool isValid = session->Initialize("/home/wxxz/workspace/weights/x_ray.onnx");
    auto* model = session->GetModel();
    std::cout << "initialize status:" << (isValid ? "true":"false") << "\n";

    cv::Mat image = cv::imread("/home/wxxz/workspace/datasets/xray/P00449.jpg");
    auto result = session->Detect(image);

    for(const auto& det : result)
    {
        std::cout << "x,y:" << det.x << " " << det.y << " w,h:"<< det.w << " " << det.h
            << " conf:" << det.confidence << " classIdx:" <<  model->labels[det.classIdx] << "\n"; 
    }

}