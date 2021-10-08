#ifndef YOLO_BASE_H_
#define YOLO_BASE_H_

#include <string>
#include <iostream>
#include <sys/stat.h>
#include <opencv2/opencv.hpp>
#include "DetectConfig.h"
//#include <common/ilogger.hpp>
//#include <common/trt_tensor.hpp>
//#include <common/object_detector.hpp>
//#include <common/cuda_tools.hpp>
//#include <common/infer_controller.hpp>
//#include <common/preprocess_kernel.cuh>
//#include <common/monopoly_allocator.hpp>
//#include <infer/trt_infer.hpp>
//#include <builder/trt_builder.hpp>
#include "app_yolo/yolo.hpp"


using namespace std;

namespace Detection
{
    class YoloBase
    {
    public:
        virtual ~YoloBase(){}

        virtual void Run() = 0;
        virtual void Run(const cv::Mat& image) = 0;
        virtual void Run(const std::vector<cv::Mat>& images) = 0;

        virtual bool CreateEngine() = 0;

        //编译模型: .onnx  ==>  .trtmodel
        virtual void Compile() = 0;

        virtual bool Require() = 0;

        virtual void SetParam(const YoloParam& param) = 0;

        virtual void GetParam(YoloParam& param) const = 0;

        virtual void Setup();

        virtual void GetBBoxResults();

    private:
        std::string m_method;


    };

}
#endif
