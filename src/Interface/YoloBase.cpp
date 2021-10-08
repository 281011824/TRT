#include "YoloBase.h"

namespace Detection
{

    //执行初始化相关的动作
    void YoloBase::Setup()
    {
       //1.检查是否有指定的.onnx模型，没有则自动下载
       bool ret = Require();
       if (!ret) return;

       //2.模型编译
       Compile();

       //3.创建引擎
       CreateEngine();

       //4.运行
       Run();
    }

}
