#include "DetectConfig.h"


using namespace std;

namespace Detection
{
    //模型精度模式
    //enum class MODEL_MODE{ FP32=0, FP16, INT8 };
    std::string GetModeName(const TRT::Mode mode)
    {
        if(mode == TRT::Mode::FP16) return "FP16";
        else if (mode == TRT::Mode::FP32) return "FP32";
        else if (mode == TRT::Mode::INT8) return "INT8";
        else return "Unknown";
    }


    bool Requires(const char* name)
    {
        auto onnx_file = iLogger::format("%s.onnx", name);
        if (not iLogger::exists(onnx_file))
        {
            INFO("Auto download %s", onnx_file.c_str());
            system(iLogger::format("wget http://zifuture.com:1556/fs/25.shared/%s", onnx_file.c_str()).c_str());
        }

        bool exists = iLogger::exists(onnx_file);
        if (not exists)
        {
            INFOE("Download %s failed", onnx_file.c_str());
        }

        return exists;
    }


//    template<class T>
//    bool IsContain(const std::vector<T>& v, const T _v)
//    {
//        if(v.empty()) return false;

//        if(std::find(v.begin(), v.end(), _v) != v.end())
//        {
//            return true;
//        }

//        return false;
//    }

}
