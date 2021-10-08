cmake_minimum_required(VERSION 3.3.0)
project(pro)
add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

set(CMAKE_CXX_STANDARD 11)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CXX_EXTENSIONS OFF)
#set(CMAKE_CXX_FLAGS -D_GLIBCXX_USE_CXX11_ABI=0)

set(CMAKE_BUILD_TYPE Debug)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/workspace)
set(HAS_PYTHON OFF)

# 如果要支持python则设置python路径
set(PythonRoot "/home/fangcheng/anaconda3/envs/track_yolov5")
set(PythonName "python3.8")

# 如果你是不同显卡，请设置为显卡对应的号码参考这里：https://developer.nvidia.com/zh-cn/cuda-gpus#compute
set(CUDA_GEN_CODE "-gencode=arch=compute_61,code=sm_61")

# 如果你的opencv找不到，可以自己指定目录
set(OpenCV_DIR   "/usr/local/opencv/opencv-4.5.3")
find_package(OpenCV 4.5.3 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

if (OpenCV_FOUND)
    message(STATUS "OpenCV library status:")
    message(STATUS "    version: ${OpenCV_VERSION}")
    message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
    message(STATUS "    libraries: ${OpenCV_LIBS}")
else ()
    message(FATAL_ERROR "Could not locate OpenCV" \n)
endif()

set(CUDA_DIR     "/usr/local/cuda-10.2")
set(CUDNN_DIR    "/usr/local/cuda-10.2/cudnn8.2.2.26")
set(TENSORRT_DIR "/home/fangcheng/TensorRT-8.0.1.6")


# set(CUDA_DIR     "/data/sxai/lean/cuda-10.2")
# set(CUDNN_DIR    "/data/sxai/lean/cudnn7.6.5.32-cuda10.2")
# set(TENSORRT_DIR "/data/sxai/lean/TensorRT-7.0.0.11")

# set(CUDA_DIR     "/data/sxai/lean/cuda-11.1")
# set(CUDNN_DIR    "/data/sxai/lean/cudnn8.2.2.26")
# set(TENSORRT_DIR "/data/sxai/lean/TensorRT-7.2.1.6")

# 因为protobuf，需要用特定版本，所以这里指定路径
set(PROTOBUF_DIR "/home/fangcheng/protobuf-3.11.4/src")

find_package(CUDA REQUIRED)

include_directories(
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/application
    #${PROJECT_SOURCE_DIR}/src/Interface
    ${PROJECT_SOURCE_DIR}/src/tensorRT
    ${PROJECT_SOURCE_DIR}/src/tensorRT/common
    ${OpenCV_INCLUDE_DIRS}/include/opencv4/opencv2
    ${PROTOBUF_DIR}/include
    ${CUDA_DIR}/include
    ${TENSORRT_DIR}/include
    ${CUDNN_DIR}/include
)

# 切记，protobuf的lib目录一定要比tensorRT目录前面，因为tensorRTlib下带有protobuf的so文件
# 这可能带来错误
link_directories(
    ${PROTOBUF_DIR}/lib
    ${TENSORRT_DIR}/lib
    ${CUDA_DIR}/lib64
    ${CUDNN_DIR}/lib
)

if("${HAS_PYTHON}" STREQUAL "ON")
    message("Usage Python ${PythonRoot}")
    include_directories(${PythonRoot}/include/${PythonName})
    link_directories(${PythonRoot}/lib)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAS_PYTHON")
endif()

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -O0 -Wfatal-errors -pthread -w -g")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -O0 -Xcompiler -fPIC -g -w ${CUDA_GEN_CODE}")
file(GLOB_RECURSE cpp_srcs ${PROJECT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE cuda_srcs ${PROJECT_SOURCE_DIR}/src/*.cu)
cuda_add_library(plugin_list SHARED ${cuda_srcs})

add_executable(${PROJECT_NAME} ${cpp_srcs})
target_link_libraries(${PROJECT_NAME} nvinfer nvinfer_plugin)
target_link_libraries(${PROJECT_NAME} cuda cublas cudart cudnn)
target_link_libraries(${PROJECT_NAME} protobuf pthread plugin_list)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

if("${HAS_PYTHON}" STREQUAL "ON")
    set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/python/trtpy)
    add_library(trtpyc SHARED ${cpp_srcs})
    target_link_libraries(trtpyc nvinfer nvinfer_plugin)
    target_link_libraries(trtpyc cuda cublas cudart cudnn)
    target_link_libraries(trtpyc protobuf pthread plugin_list)
    target_link_libraries(trtpyc ${OpenCV_LIBS})
    target_link_libraries(trtpyc "${PythonName}")
    target_link_libraries(${PROJECT_NAME} "${PythonName}")
endif()

add_custom_target(
    yolo                                                 # OUTPUT：目标文件名称，代表下方的命令；
    DEPENDS ${PROJECT_NAME}                              # DEPENDS：执行命令时需要的依赖；
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/workspace
    COMMAND ./${PROJECT_NAME} yolo                       # COMMAND：需要执行的命令；
)

add_custom_target(
    yolo_fast
    DEPENDS ${PROJECT_NAME}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/workspace
    COMMAND ./${PROJECT_NAME} yolo_fast
)

add_custom_target(
    centernet
    DEPENDS ${PROJECT_NAME}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/workspace
    COMMAND ./${PROJECT_NAME} centernet
)

add_custom_target(
    alphapose 
    DEPENDS ${PROJECT_NAME}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/workspace
    COMMAND ./${PROJECT_NAME} alphapose
)

add_custom_target(
    retinaface
    DEPENDS ${PROJECT_NAME}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/workspace
    COMMAND ./${PROJECT_NAME} retinaface
)

add_custom_target(
    dbface
    DEPENDS ${PROJECT_NAME}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/workspace
    COMMAND ./${PROJECT_NAME} dbface
)

add_custom_target(
    arcface 
    DEPENDS ${PROJECT_NAME}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/workspace
    COMMAND ./${PROJECT_NAME} arcface
)

add_custom_target(
    fall
    DEPENDS ${PROJECT_NAME}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/workspace
    COMMAND ./${PROJECT_NAME} fall_recognize
)

add_custom_target(
    scrfd
    DEPENDS ${PROJECT_NAME}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/workspace
    COMMAND ./${PROJECT_NAME} scrfd
)

add_custom_target(
    lesson
    DEPENDS ${PROJECT_NAME}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/workspace
    COMMAND ./${PROJECT_NAME} lesson
)

add_custom_target(
    pyscrfd
    DEPENDS trtpyc
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/python
    COMMAND python test_scrfd.py
)

add_custom_target(
    pyinstall
    DEPENDS trtpyc
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/python
    COMMAND python setup.py install
)

add_custom_target(
    pytorch
    DEPENDS trtpyc
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/python
    COMMAND python test_torch.py
)

add_custom_target(
    pycenternet
    DEPENDS trtpyc
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/python
    COMMAND python test_centernet.py
)
