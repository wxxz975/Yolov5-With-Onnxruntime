

#include "yolov5/Yolov5Session.h"

#include <iostream>
#include <filesystem>
#include <chrono>
#include <string>

#include "Mics.h"

int main(int argc, char* argv[])
{
    const std::string imageExt = ".jpg"; // 图像的扩展名
    bool renderAndSave = true; // 是否绘制外框
    if(argc != 2)
    {
        std::cout << "Usage: " << argv[0] << " inputImagePath" << "\n";
        return 0;
    }   

    ISession *session = new Yolov5Session();
    bool isValid = session->Initialize("/home/wxxz/workspace/weights/x_ray.onnx");
    auto* model = session->GetModel();
    std::cout << "initialize status:" << (isValid ? "true":"false") << "\n";

    std::filesystem::path data;
    
    data = argv[1]; // 获取图像的完整地址
    std::vector<std::string> filenames;
    if(std::filesystem::is_directory(data))
    {
        for(const auto& entry : std::filesystem::directory_iterator(data))
        {
            if(entry.is_regular_file() && entry.path().extension() == imageExt)
            {
                filenames.push_back(entry.path().string());
            }
        }
    }else if(data.extension().string() == imageExt)
    {
        filenames.push_back(data.string());
    }else 
        return 0;




    for(const auto& imagePath : filenames)
    {
        cv::Mat image = cv::imread(imagePath);
        
        auto start_time = std::chrono::high_resolution_clock::now();
        auto result = session->Detect(image);
        auto end_time = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        std::cout << "运行时间: " << duration.count() << " 毫秒" << "\n";

        for(const auto& det : result)
        {
            std::cout << "x,y:" << det.x << " " << det.y << " w,h:"<< det.w << " " << det.h
                << " conf:" << det.confidence << " classIdx:" <<  model->labels[det.classIdx] << "\n"; 
        }

        if(renderAndSave)
        {
            cv::Mat out = RenderBoundingBoxes(image, result, model->labels);
            std::filesystem::path oriPath = imagePath;
            std::string newPath = std::filesystem::current_path().string() + "/result/"  + oriPath.filename().string();

            cv::imwrite(newPath, out);
        }
    }
    

    return 0;
}