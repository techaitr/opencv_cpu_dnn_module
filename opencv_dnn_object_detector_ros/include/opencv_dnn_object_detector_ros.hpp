#ifndef __OPENCV_DNN_OBJECT_DETECTOR_ROS__
#define __OPENCV_DNN_OBJECT_DETECTOR_ROS__
#include <ros/ros.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv4/opencv2/dnn.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/dnn/dnn.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <string>
#include <limits>
#include <cmath>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <iostream>
#include <iomanip>
#include <bits/stdc++.h>
#include <math.h>
#include <ros/console.h>
#include <atomic>
#include <ros/package.h>

float CONFIDENCE_THRESHOLD = 0.5; // Confidence threshold constexpr
float NMS_THRESHOLD = 0.4;        // Non-maximum suppression threshold - 0.4 constexpr
int NUM_CLASSES = 80;   // Number of classes - 80

// colors for bounding boxes
const cv::Scalar colors[] = {
    {0, 255, 255},
    {255, 255, 0},
    {0, 255, 0},
    {255, 0, 0}};

const auto NUM_COLORS = sizeof(colors) / sizeof(colors[0]);

class opencv_dnn_object_detector_ros
{
private:
    /* data */
    bool initialization();
    std::string frozenInferenceGraphName = "frozen_inference_graph.pb";
    std::string modelTensorflowPbtxtPath;
    std::string modelTensorflowPbtxtName = "ssd_mobilenetSinan.pbtxt";

    std::string frozenInferenceGraphPath;
    std::vector<std::string> classNames;
    std::string formatName = "darknet"; //darknet veya tensorflow olabilir.
    std::string packagePath;
    std::string modelClassName = "coco.names";
    std::string modelCfgName = "yolov4-tiny.cfg";
    std::string modelWeightsName = "yolov4-tiny.weights";
    std::string COCONamesFilePath;
    std::string cfgFilePath = "/input/cfg/";
    std::string weightsFilePath = "/input/";
    std::string line;
    int initialConf = (int)(CONFIDENCE_THRESHOLD * 100);
    int initialNMS = (int)(NMS_THRESHOLD * 100);
    void infiniteDarknet();
    void infiniteTensorflow();
    cv::dnn::Net net;
    std::vector<std::string> outputNames;
    double inference_fps = 0;
    double total_fps = 0;
    cv::Mat frame, blob;
    std::vector<cv::Mat> detections;
    cv::VideoCapture cap;
    bool stopDetection = false;
    ros::NodeHandle nh;
    bool readFromYAMLFile();

public:
    int inpWidth;  // Width of network's input image - 416
    int inpHeight; // Height of network's input image - 608
    int count = 0;
    opencv_dnn_object_detector_ros(/* args */);
    ~opencv_dnn_object_detector_ros();
    bool readFilesYOLOV3();
    void cleanup();
    void reset(bool);
    bool openVideoCaptureStream();
    bool setupNet();
    void setCap(cv::VideoCapture &);
    cv::VideoCapture getCap();
    std::vector<cv::Rect> detectDarknet(const cv::Mat &);
    std::vector<cv::Rect> detectTensorflow(const cv::Mat &);
};

#endif