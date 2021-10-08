#include <iostream>
#include <string.h>
#include <functional>
#include "Interface/DetectConfig.h"
#include "Interface/DetectorFactory.h"
#include "Interface/YoloBase.h"
#include "Interface/Yolox.h"
#include "Interface/Yolov5.h"

using namespace std;

int main(int argc, char** argv)
{
    //模型选择
    std::string method("yolov5s");
    if(argc > 1)
    {
        method = argv[1];
    }

    //模型数据精度
    std::string mode("FP32");
    if(argc > 2)
    {
        mode = argv[2];
    }

    //检测是文件：  ["IMG", "VIDEO", "CAMERA"]
    std::string type("IMG");
    if(argc > 3)
    {
        type = argv[3];
    }


    // =================== YoloX ==================== //
    //std::shared_ptr<Detection::YoloFactory> pYoloFactory = std::make_shared<Detection::YoloFactory>();
    Detection::YoloFactory* pYoloFactory = new Detection::YoloxFactory();

    //std::shared_ptr<Detection::YoloBase> pYoloX = std::make_shared<Detection::YoloBase>(pYoloFactory->createYolo());
    Detection::YoloBase* pYoloX = pYoloFactory->createYolo();

    Detection::YoloParam yoloxParam(method, TRT::Mode::FP32);
    pYoloX->SetParam(yoloxParam);

    pYoloX->Setup();

    // 释放资源
    delete pYoloX;
    delete pYoloFactory;


    // =================== Yolov5 ==================== //
//    Detection::YoloFactory* pYolov5Factory = new Detection::Yolov5Factory();

//    Detection::YoloBase* pYolov5 = pYolov5Factory->createYolo();

//    Detection::YoloParam yolov5Param(method, TRT::Mode::FP32);
//    pYolov5->SetParam(yolov5Param);

//    pYolov5->Setup();

//    // 释放资源
//    delete pYolov5;
//    delete pYolov5Factory;


    return 0;
}
