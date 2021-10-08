#ifndef YOLO_X_H_
#define YOLO_X_H_

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "YoloBase.h"
#include "DetectConfig.h"
#include "common/trt_tensor.hpp"
#include "builder/trt_builder.hpp"
#include "app_yolo/yolo.hpp"


namespace Detection
{

    class YoloX : public YoloBase
    {
    public:
        //YoloX();
        //~YoloX() override;

        //YoloX(const YoloParam& param);

        void Run() override;
        void Run(const cv::Mat& image) override;
        void Run(const std::vector<cv::Mat>& vecImage) override;

        //设定参数
        //void SetParam(const YoloParam& param) override;

        //编译模型: .onnx  ==>  .trtmodel
        virtual void Compile() override;

        virtual bool Require() override;

        virtual void SetParam(const YoloParam& param) override;

        virtual void GetParam(YoloParam& param) const override;

        //创建引擎
        virtual bool CreateEngine() override;

        void SetImage(const cv::Mat& image);

        //void GetResults();

        void SetImages(const std::vector<cv::Mat>& images);

    private:

        cv::Mat m_image;

        std::vector<cv::Mat> m_images;

        YoloParam m_param;

        std::shared_ptr<Yolo::Infer> m_pEngine;
    };

}

#endif
