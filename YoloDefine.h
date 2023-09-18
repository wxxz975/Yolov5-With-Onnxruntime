#pragma once


typedef struct _DetectionResultNode
{
    float x, y, w, h;
    int classIdx;
    float confidence;
}ResultNode, *pResultNode;



struct RawResult
{
    float cx;       // center x
    float cy;       // center y
    float w;        // width
    float h;        // height
    float cls_conf; // class confidence
}__attribute__((packed));

typedef RawResult* pRawResult;

#define YOLOV5_OUTBOX_ELEMENT_COUNT 5

