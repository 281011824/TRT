#include "DetectorFactory.h"



//static void test(Yolo::Type type, TRT::Mode mode, const string& model)
//{
//    int deviceid = 0;
//    auto mode_name = TRT::mode_string(mode);
//    TRT::set_device(deviceid);

//    //lamba 函数，用于处理int8时的数据标定
//    auto int8process = [=](int current, int count, const vector<string>& files, shared_ptr<TRT::Tensor>& tensor)
//    {
//        INFO("Int8 %d / %d", current, count);
//        for(int i = 0; i < files.size(); ++i)
//        {
//            auto image = cv::imread(files[i]);
//            Yolo::image_to_tensor(image, tensor, type, i);
//        }
//    };

//    const char* name = model.c_str();
//    if(!Require(name)) return;

//    string onnx_file = iLogger::format("%s.onnx", name);
//    string model_file = iLogger::format("%s.%s.trtmodel", name, mode_name);
//    int test_batch_size = 16;
//    if(not iLogger::exists(model_file))
//    {
//        TRT::compile(
//            mode,                     // FP32、FP16、INT8
//            test_batch_size,          // max batch size
//            onnx_file,                // source
//            model_file,               // save to
//            {},
//            int8process,
//            "inference"
//        );
//    }

//    //inference_and_performance(deviceid, model_file, mode, type, name);
//    inference_camera(deviceid, model_file, mode, type, name);
//}
