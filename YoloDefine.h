#pragma once


typedef struct _DetectionResultNode
{
    float x, y, w, h;
    int classIdx;
    float confidence;
}ResultNode, *pResultNode;


typedef struct RawResult
{
    
}RawResult, *pRawResult;

