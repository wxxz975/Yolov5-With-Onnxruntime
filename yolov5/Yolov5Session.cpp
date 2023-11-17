#include "Yolov5Session.h"


Yolov5Session::Yolov5Session()
{

}


Yolov5Session::~Yolov5Session()
{
    if(processor_)
        delete processor_;
    
    if(model_)
        delete model_;

    processor_ = nullptr;
    model_ = nullptr;
}

bool Yolov5Session::Initialize(const std::string& modelPath)
{
    
    return CreateSession(modelPath) && ParseModel() && WarmUpModel();
}

std::vector<ResultNode> Yolov5Session::Detect(const cv::Mat& image)
{
    std::vector<ResultNode> result;
    const auto& inputNames = model_->inputNamesPtr;
    const auto& outputNames = model_->outputNamesPtr;
    
    if(processor_)
    {
        auto inputTensor = processor_->Preprocess(image);

        std::vector<Ort::Value>outTensor = session_.Run(Ort::RunOptions{nullptr}, inputNames.data(), inputTensor.data(), inputTensor.size(), outputNames.data(), outputNames.size());
        
        result = processor_->Postprocess(outTensor, image.size(), confidenceThreshold_, iouThreshold_);
    }

    return result;
}

bool Yolov5Session::CreateSession(const std::filesystem::path& modelPath)
{
    if(!std::filesystem::exists(modelPath))
        return false;

    envName_ = modelPath.filename().string();
    env_ = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, envName_.c_str());
    
    sessionOpt = Ort::SessionOptions();
    sessionOpt.SetIntraOpNumThreads(0);
    sessionOpt.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);

    if(useGpu && IsGPUAvailable())
    {
        auto cudaOptions = CreateCudaOptions();
        sessionOpt.AppendExecutionProvider_CUDA(cudaOptions);
    }

    session_ = Ort::Session(env_, modelPath.c_str(), sessionOpt);
    return true;
}

bool Yolov5Session::ParseModel()
{
    model_ = ModelParser::parse(&session_);
    if(!model_)
        return false;
    processor_ = new ModelProcessor(model_);

    return true;
}


OrtCUDAProviderOptions Yolov5Session::CreateCudaOptions()
{
    OrtCUDAProviderOptions cudaOption;
    cudaOption.device_id = 0;
    cudaOption.arena_extend_strategy = 0;
    cudaOption.gpu_mem_limit = std::numeric_limits<size_t>::max();
    cudaOption.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchDefault;
    cudaOption.do_copy_in_default_stream = 1;

    return cudaOption;
}

bool Yolov5Session::IsGPUAvailable()
{
    std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
    auto cudaAvailable = std::find(availableProviders.begin(), availableProviders.end(),
        "CUDAExecutionProvider");

    return cudaAvailable != availableProviders.end();
}

template<class T>
static size_t MultiVec(const std::vector<T>& data)
{
    size_t total = 1;
    for(const auto& it : data)
        total *= it;
    
    return total;
}

bool Yolov5Session::WarmUpModel()
{
    if(!warmup) 
        return true;

    Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemTypeDefault);
    const auto& inputName = model_->inputNamesPtr;
    const auto& outputName = model_->outputNamesPtr;
    const auto& inputShape =  model_->inputShapes;
    size_t inputSize = MultiVec(inputShape.at(0));

    std::vector<float> input_data(inputSize);


    std::vector<Ort::Value> fetches;
    fetches.push_back(
        Ort::Value::CreateTensor<float>(memInfo, input_data.data(), input_data.size(), inputShape[0].data(), inputShape[0].size())
    );

    
    session_.Run(Ort::RunOptions{nullptr}, inputName.data(), fetches.data(), fetches.size(), outputName.data(), outputName.size());

    return true;
}