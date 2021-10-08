#ifndef YOLO_V5_H_
#define YOLO_V5_H_

#include <string>
#include "YoloBase.h"
#include "common/trt_tensor.hpp"
#include "builder/trt_builder.hpp"
#include <common/ilogger.hpp>
#include <infer/trt_infer.hpp>



namespace Detection
{

    class Yolov5:public YoloBase
    {
    public:
        //Yolov5();
        //~Yolov5() override;

        void Run() override;
        void Run(const cv::Mat& image) override;
        void Run(const std::vector<cv::Mat>& vecImage) override;

        virtual bool CreateEngine() override;

        //编译模型: .onnx  ==>  .trtmodel
        virtual void Compile() override;

        virtual bool Require() override;

        virtual void SetParam(const YoloParam& param) override;

        virtual void GetParam(YoloParam& param) const override;

    private:
        // X, S, M, L, etc.
        YOLOV5_TYPE m_type;

        YoloParam m_param;

        std::shared_ptr<Yolo::Infer> m_pEngine;

    };


}

#endif
