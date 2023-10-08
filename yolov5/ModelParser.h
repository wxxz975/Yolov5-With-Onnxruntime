#pragma once
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "Model.h"


class ModelParser
{
private:
    /* data */
public:
    ModelParser() = default;
    ~ModelParser() = default;

    static Model* parse(Ort::Session* session);

private:
    static bool parseInput(Ort::Session* session, Model* model);

    static bool parseOutput(Ort::Session* session, Model* model);

    static bool parseLabels(Ort::Session* session, Model* model, const std::string& labelKey = "names");

    static std::vector<std::string> parseLabelsRaw(const std::string& rawData);

};

