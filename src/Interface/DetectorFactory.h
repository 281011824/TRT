#ifndef DETECTOR_FACTORY_H_
#define DETECTOR_FACTORY_H_

#include <string>
#include <iostream>
#include "YoloBase.h"
#include "Yolox.h"
#include "Yolov5.h"


using namespace std;

//抽象工厂设计模式  +  模板方法设计模式
//1.使用抽象工厂设计模式解决统一对外主接口；
//2.模板方法设计模式解决统一流程
namespace Detection
{

    class YoloFactory
    {
    public:
        virtual YoloBase * createYolo() = 0;
        virtual ~YoloFactory(){}
    };

    class YoloxFactory : public YoloFactory
    {
    public:
        virtual ~YoloxFactory(){};
        YoloBase * createYolo()
        {
            return new YoloX();
        }
    };

    class Yolov5Factory : public YoloFactory
    {
    public:
        virtual ~Yolov5Factory(){};
        YoloBase * createYolo()
        {
            return new Yolov5();
        }
    };


}

#endif
