#ifndef DETECT_CONFIG_H_
#define DETECT_CONFIG_H_

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <builder/trt_builder.hpp>
#include <common/preprocess_kernel.cuh>


using namespace std;

namespace Detection
{
    const vector<string> YOLO_LIST = {"yolox_x", "yolox_l", "yolox_m", "yolox_s","yolov5x6", "yolov5l6" ,
                                      "yolov5m6", "yolov5s6", "yolov5x", "yolov5l", "yolov5m", "yolov5s"};

    const std::vector<std::string> YOLOX_LIST = {"yolox_x", "yolox_l", "yolox_m", "yolox_s"};

    const std::vector<std::string> YOLOV5_LIST = {"yolov5s", "yolov5s6", "yolov5m", "yolov5m6",
                                                  "yolov5x", "yolov5x6", "yolov5l6", "yolov5l"};

    const char* DetectLabels[] =
    {
        "person", "bicycle", "car", "motorcycle", "airplane",
        "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
        "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
        "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
        "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
        "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
        "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
        "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
        "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"
    };

    //yolo类型的枚举
    enum class YOLO_TYPE { YoloV3=0, YoloV4, YoloV5, YoloR, YoloX };

    enum class YOLOV5_TYPE {S=0, S6, M, M6, L, L6 };

    enum class YOLOX_TYPE {S=0, M, L, X};

    const int MAX_IMAGE_BBOX  = 1024;
    const int NUM_BOX_ELEMENT = 7;      // left, top, right, bottom, confidence, class, keepflag

    template<class T>
    bool IsContain(const std::vector<T>& v, const T _v)
    {
        if(v.empty()) return false;

        if(std::find(v.begin(), v.end(), _v) != v.end())
        {
            return true;
        }

        return false;
    }


    std::string GetModeName(const TRT::Mode mode);
    bool Requires(const char* name);

    struct YoloParam
    {
        string method;

        TRT::Mode mode;

        string modeName;

        YOLO_TYPE type;

        int deviceId;

        int maxBatchSize;

        string onnxFile;

        string trtModelFile;

        float score;

        float nmsThreshold;

        YoloParam():method("Yolov5"), mode(TRT::Mode::FP32), modeName(""), type(YOLO_TYPE::YoloV5), deviceId(0), maxBatchSize(8),
            onnxFile(""), trtModelFile(""), score(0.5f), nmsThreshold(0.4f){}

        YoloParam(const string _method, const TRT::Mode _mode=TRT::Mode::FP32, const int _deviceId=0,const int _batcSize=8,
                const float _score=0.5f, const float _nmsThreshold=0.4f)
            :method(_method),  mode(_mode), deviceId(_deviceId), maxBatchSize(_batcSize), score(_score), nmsThreshold(_nmsThreshold)
        {
            modeName = GetModeName(mode);
            bool ret = IsContain<std::string>(YOLO_LIST, method);
            if(ret)
            {
                onnxFile     = method + "." + modeName + ".onnx";
                trtModelFile = method + "." + modeName + ".trtmodel";
            }

            if(IsContain<std::string>(YOLOX_LIST, method))
            {
                type = YOLO_TYPE::YoloX;
            }
            else if(IsContain<std::string>(YOLOV5_LIST, method))
            {
                type = YOLO_TYPE::YoloV5;
            }

        }

        YoloParam& operator=(const YoloParam& obj)
        {
            this->method   = obj.method;
            this->onnxFile = obj.onnxFile;
            this->trtModelFile = obj.trtModelFile;
            this->deviceId = obj.deviceId;
            this->mode     = obj.mode;
            this->type     = obj.type;
            this->score    = obj.score;
            this->nmsThreshold = obj.nmsThreshold;
            return *this;
        }
    };

    //左上角坐标和长度与宽度的BBox形式：(x, y, h, w)
    template <class T>
    struct BBoxXYHW
    {
        T x;
        T y;
        T h;
        T w;

        BBoxXYHW():x(0), y(0), h(0), w(0){}
        BBoxXYHW(const T _x, const T _y, const T _h, const T _w)
            :x(_x), y(_y), h(_h), w(_w){}
        BBoxXYHW& operator=(const BBoxXYHW& b)
        {
            this->x = b.x;
            this->y = b.y;
            this->h = b.h;
            this->w = b.w;
            return *this;
        }
    };
    typedef BBoxXYHW<int> BBoxXyhwI;
    typedef BBoxXYHW<float> BBoxXyhwF;
    typedef BBoxXYHW<double> BBoxXyhwD;

    //左上角坐标和右下角坐标的BBox形式：(x0, y0, x1, y1)
    template <class T>
    struct BBoxXYXY
    {
        T x0;
        T y0;
        T x1;
        T y1;

        BBoxXYXY():x0(0), y0(0), x1(0), y1(0){}
        BBoxXYXY(const T _x0, const T _y0, const T _x1, const T _y1)
            :x0(_x0), y0(_y0), x1(_x1), y1(_y1){}

        BBoxXYXY& operator=(const BBoxXYXY& b)
        {
            this->x0 = b.x0;
            this->y0 = b.y0;
            this->x1 = b.x1;
            this->y1 = b.y1;
            return *this;
        }
    };
    typedef BBoxXYXY<int> BBoxXyxyI;
    typedef BBoxXYXY<float> BBoxXyxyF;
    typedef BBoxXYXY<double> BBoxXyxyD;

    //Yolo使用的边界框格式
    struct YoloBBox
    {
        float score;
        int label;
        BBoxXyxyI bbox;

        YoloBBox() = default;
        YoloBBox(const float _score, const int _label, const BBoxXyxyI& _bbox)
            :score(_score), label(_label), bbox(_bbox)
        {}

        YoloBBox& operator=(const YoloBBox& obj)
        {
            this->score = obj.score;
            this->label = obj.label;
            this->bbox = obj.bbox;
            return *this;
        }
    };




}

#endif
