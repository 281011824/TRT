#include "Yolov5.h"


namespace Detection
{

    void Yolov5::Run()
    {
        //获取摄像机接口
        cv::VideoCapture cam(0);    //启动相机的端口号，一般默认相机为video0，可以通过`ls /dev/video*`来查看

        //检测接口是否打开
        if (!cam.isOpened())
        {
            return;
        }

        printf("摄像头开启正常\n");
        //声明图像帧
        cv::Mat srcframe;
        cv::namedWindow("Camera_Detection", cv::WINDOW_NORMAL);
        std::vector<shared_future<Yolo::BoxArray>> boxes_array;
        //cv::Mat src_img = cv::imread("/mnt/code/tensorRT_cpp-main/workspace/inference/car.jpg");
        while(true)
        {
            //从摄像头读取图像
            cam >> srcframe;

            //计时器开始
            auto begin_timer = iLogger::timestamp_now_float();

            //提交一贞图像
            std::shared_future<Yolo::BoxArray> boxes_list = m_pEngine->commit(srcframe);

            //等待推理完成
            boxes_list.wait();

            float inference_time = (iLogger::timestamp_now_float() - begin_timer);

            //绘制检测结果图
            auto image = srcframe.clone();
            auto boxes  = boxes_list.get();
            for(auto& obj : boxes)
            {
                uint8_t b, g, r;
                tie(b, g, r) = iLogger::random_color(obj.class_label);
                cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom), cv::Scalar(b, g, r), 5);

                auto name    = DetectLabels[obj.class_label];
                auto caption = iLogger::format("%s %.2f", name, obj.confidence);
                int width    = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 3;
                cv::rectangle(image, cv::Point(obj.left-3, obj.top-33), cv::Point(obj.left + width, obj.top), cv::Scalar(b, g, r), -1);
                cv::putText(image, caption, cv::Point(obj.left, obj.top-5), 0, 1, cv::Scalar::all(0), 2, 16);
                cv::putText(image, std::to_string(inference_time) + " ms", cv::Point(10, 20), 0, 1, cv::Scalar(128, 255, 0), 1, 8);
            }

            cv::imshow("Camera_Detection", image);
            cv::waitKey(5);

            //睡眠5毫秒
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }

        cv::destroyWindow("Camera_Detection");
        cam.release();
    }

    void Yolov5::Run(const cv::Mat& img)
    {
        if(img.empty())
        {
            return;
        }
    }

    void Yolov5::Run(const std::vector<cv::Mat>& vecImg)
    {
        if(vecImg.empty())
        {
            return;
        }

    }

    bool Yolov5::CreateEngine()
    {
        m_pEngine = Yolo::create_infer(m_param.trtModelFile, Yolo::Type::V5, 0, m_param.score, m_param.nmsThreshold);
        if(nullptr == m_pEngine)
        {
            INFOE("Engine is nullptr");
            return false;
        }

        return true;
    }


    void Yolov5::Compile()
    {
        auto mode_name = GetModeName(m_param.mode);  // "FP32"
        TRT::set_device(m_param.deviceId);

        auto int8process = [=](int current, int count, const vector<string>& files, shared_ptr<TRT::Tensor>& tensor)
        {
            INFO("Int8 %d / %d", current, count);
            for(int i = 0; i < files.size(); ++i)
            {
                auto image = cv::imread(files[i]);
                Yolo::image_to_tensor(image, tensor, Yolo::Type::V5, i);
            }
        };

        INFO("===================== test %s %s =========================", mode_name.c_str(), m_param.method.c_str());

        if(not iLogger::exists(m_param.trtModelFile))
        {
            TRT::compile(
                m_param.mode,             // FP32、FP16、INT8
                m_param.maxBatchSize,     // max batch size
                m_param.onnxFile,         // source
                m_param.trtModelFile,     // save to
                {},
                int8process,
                "inference"
            );
        }
    }

    bool Yolov5::Require()
    {
        bool ret = Requires(m_param.method.c_str());
        return ret;
    }

    void Yolov5::SetParam(const YoloParam& param)
    {
        m_param = param;
    }

    void Yolov5::GetParam(YoloParam& param) const
    {
        param = m_param;
    }

}
