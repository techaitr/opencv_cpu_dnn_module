#include "opencv_dnn_object_detector_ros.hpp"

opencv_dnn_object_detector_ros::opencv_dnn_object_detector_ros(/* args */)
{
    initialization();
}

bool opencv_dnn_object_detector_ros::initialization()
{

    packagePath = ros::package::getPath("opencv_dnn_object_detector_ros");
    readFromYAMLFile();
    return true;

    COCONamesFilePath = packagePath + "/input/data/" + modelClassName;
    if (formatName == "darknet")
    {
        cfgFilePath = packagePath + "/input/cfg/" + modelCfgName;
        weightsFilePath = packagePath + "/input/" + modelWeightsName;
        std::cout << "packagePath: " << packagePath << std::endl;
    }
    else if (formatName == "tensorflow")
    {
        frozenInferenceGraphPath = packagePath + "/input/" + frozenInferenceGraphName;
        modelTensorflowPbtxtPath = packagePath + "/input/" + modelTensorflowPbtxtName;
    }
    readFilesYOLOV3();
    setupNet();
    openVideoCaptureStream();
}
bool opencv_dnn_object_detector_ros::readFromYAMLFile()
{
    std::cout << "readFromYAMLFile is called." << std::endl;

    nh.getParam("/formatName", formatName);
    nh.getParam("/modelClassName", modelClassName);
    nh.getParam("/CONFIDENCE_THRESHOLD", CONFIDENCE_THRESHOLD);
    nh.getParam("/NMS_THRESHOLD", NMS_THRESHOLD);
    nh.getParam("/NUM_CLASSES", NUM_CLASSES);
    nh.getParam("/stopDetection", stopDetection);

    if (formatName == "darknet")
    {
        nh.getParam("/darknet/inputWidthYOLO", inpWidth);
        nh.getParam("/darknet/inputHeightYOLO", inpHeight);
    }
    else if (formatName == "tensorflow")
    {
        nh.getParam("/tensorflow/inputWidthTensorflow", inpWidth);
        nh.getParam("/tensorflow/inputHeightTensorflow", inpHeight);
    }
    std::cout << "formatName: " << formatName << std::endl;
    std::cout << "inpWidth: " << inpWidth << std::endl;
    std::cout << "inpHeight: " << inpHeight << std::endl;
        std::cout << "NMS_THRESHOLD: " << NMS_THRESHOLD << std::endl;
        std::cout << "modelClassName: " << modelClassName << std::endl;

}
opencv_dnn_object_detector_ros::~opencv_dnn_object_detector_ros()
{
    cap.release();
    cv::destroyAllWindows();
}
bool opencv_dnn_object_detector_ros::readFilesYOLOV3()
{
    try
    {
        if (formatName == "darknet" || formatName == "tensorflow")
        {
            std::cout << "readfilesYOLOV3 is called." << std::endl;
            std::ifstream class_file(COCONamesFilePath);
            if (!class_file)
            {
                std::cerr << "failed to open" << COCONamesFilePath << "\n";
                return false;
            }
            while (getline(class_file, line))
            {
                classNames.push_back(line);
            }
            return true;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
        return false;
    }
}
bool opencv_dnn_object_detector_ros::setupNet()
{
    try
    {
        if (formatName == "darknet")
        {
            net = cv::dnn::readNetFromDarknet(cfgFilePath, weightsFilePath);
            outputNames = net.getUnconnectedOutLayersNames();
        }
        else if (formatName == "tensorflow")
        {
            /* code */
            net = cv::dnn::readNet(frozenInferenceGraphPath, modelTensorflowPbtxtPath, "TensorFlow");
            outputNames = net.getUnconnectedOutLayersNames();
            for (size_t i = 0; i < outputNames.size(); i++)
            {
                std::cout << i << " : " << outputNames.at(i) << std::endl;
            }
        }

        return true;
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
        return false;
    }
}

std::vector<cv::Rect> opencv_dnn_object_detector_ros::detectDarknet(const cv::Mat &frame)
{
    std::vector<cv::Rect> boundingBox;

    if (!stopDetection)
    {
        if (!frame.empty())
        {
            auto total_start = std::chrono::steady_clock::now();
            cv::dnn::blobFromImage(frame, blob, 0.00392, cv::Size(inpWidth, inpHeight), cv::Scalar(), true, false, CV_32F);
            net.setInput(blob);
            auto dnn_start = std::chrono::steady_clock::now();
            net.forward(detections, outputNames);
            auto dnn_end = std::chrono::steady_clock::now();

            std::vector<int> indices[NUM_CLASSES];
            std::vector<cv::Rect> boxes[NUM_CLASSES];
            std::vector<float> scores[NUM_CLASSES];

            for (auto &output : detections)
            {
                const auto num_boxes = output.rows;
                for (int i = 0; i < num_boxes; i++)
                {

                    auto x = output.at<float>(i, 0) * frame.cols;
                    auto y = output.at<float>(i, 1) * frame.rows;
                    auto width = output.at<float>(i, 2) * frame.cols;
                    auto height = output.at<float>(i, 3) * frame.rows;
                    cv::Rect rect(x - width / 2, y - height / 2, width, height);

                    for (int c = 0; c < NUM_CLASSES; c++)
                    {
                        auto confidence = *output.ptr<float>(i, 5 + c);
                        if (confidence >= CONFIDENCE_THRESHOLD)
                        {
                            boxes[c].push_back(rect);
                            scores[c].push_back(confidence);
                        }
                    }
                }
            }

            for (int c = 0; c < NUM_CLASSES; c++)
            {
                cv::dnn::NMSBoxes(boxes[c], scores[c], 0.0, NMS_THRESHOLD, indices[c]);
            }

            for (int c = 0; c < NUM_CLASSES; c++)
            {
                for (size_t i = 0; i < indices[c].size(); ++i)
                {

                    if (classNames[c] == "person")
                    {
                        const auto color = colors[c % NUM_COLORS];

                        auto idx = indices[c][i];
                        const auto &rect = boxes[c][idx];
                        rectangle(frame, cv::Point(rect.x, rect.y), cv::Point(rect.x + rect.width, rect.y + rect.height), color, 3);

                        std::ostringstream label_ss;
                        label_ss << classNames[c] << ": " << std::fixed << std::setprecision(2) << scores[c][idx];
                        auto label = label_ss.str();

                        int baseline;
                        auto label_bg_sz = cv::getTextSize(label.c_str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);
                        rectangle(frame, cv::Point(rect.x, rect.y - label_bg_sz.height - baseline - 10), cv::Point(rect.x + label_bg_sz.width, rect.y), color, cv::FILLED);
                        putText(frame, label.c_str(), cv::Point(rect.x, rect.y - baseline - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 0));

                        boundingBox.push_back(rect);
                    }
                }
            }

            auto total_end = std::chrono::steady_clock::now();

            inference_fps = 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(dnn_end - dnn_start).count();
            total_fps = 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();
            std::ostringstream stats_ss;
            stats_ss << std::fixed << std::setprecision(2);
            stats_ss << "Inference FPS: " << inference_fps << ", Total FPS: " << total_fps;
            auto stats = stats_ss.str();

            int baseline;
            auto stats_bg_sz = getTextSize(stats.c_str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);
            rectangle(frame, cv::Point(0, 0), cv::Point(stats_bg_sz.width, stats_bg_sz.height + 10), cv::Scalar(0, 0, 0), cv::FILLED);
            putText(frame, stats.c_str(), cv::Point(0, stats_bg_sz.height + 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 255, 255));

            std::ostringstream stats_ss2;
            stats_ss2 << std::fixed << std::setprecision(2);
            stats_ss2 << "Detected Person: " << boundingBox.size();
            auto stats2 = stats_ss2.str();
            auto stats_bg_sz2 = getTextSize(stats2.c_str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);
            rectangle(frame, cv::Point(0, stats_bg_sz.height + 10), cv::Point(stats_bg_sz2.width, stats_bg_sz2.height + 30), cv::Scalar(0, 0, 0), cv::FILLED);
            putText(frame, stats2.c_str(), cv::Point(0, stats_bg_sz2.height + 25), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 255, 255));

            imshow("kWinName", frame);

            if ((char)cv::waitKey(1) == 27)
            {
                std::cout << "ESC pressed" << std::endl;
                cap.release();
                cv::destroyAllWindows();

                reset(true);
            }
        }
    }

    std::cout << "Detected Person:" << boundingBox.size() << std::endl;
    return boundingBox;
}

std::vector<cv::Rect> opencv_dnn_object_detector_ros::detectTensorflow(const cv::Mat &frame)
{
    std::vector<cv::Rect> boundingBox;

    auto total_start = std::chrono::steady_clock::now();

    // Create a blob from the image
    cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0, cv::Size(inpWidth, inpHeight), cv::Scalar(127.5, 127.5, 127.5),
                                          true, false);

    // Set the blob to be input to the neural network
    net.setInput(blob);

    // Forward pass of the blob through the neural network to get the predictions
    cv::Mat output = net.forward();

    // Matrix with all the detections
    cv::Mat results(output.size[2], output.size[3], CV_32F, output.ptr<float>());

    // Run through all the predictions
    for (int i = 0; i < results.rows; i++)
    {
        int class_id = int(results.at<float>(i, 1));
        float confidence = results.at<float>(i, 2);

        // Check if the detection is over the min threshold and then draw bbox
        if (confidence > CONFIDENCE_THRESHOLD)
        {
            int bboxX = int(results.at<float>(i, 3) * frame.cols);
            int bboxY = int(results.at<float>(i, 4) * frame.rows);
            int bboxWidth = int(results.at<float>(i, 5) * frame.cols - bboxX);
            int bboxHeight = int(results.at<float>(i, 6) * frame.rows - bboxY);
            cv::Rect rect(cv::Point2i(bboxX, bboxY), cv::Point2i(bboxWidth, bboxHeight));
            boundingBox.push_back(rect);

            rectangle(frame, cv::Point(bboxX, bboxY), cv::Point(bboxX + bboxWidth, bboxY + bboxHeight), cv::Scalar(0, 0, 255), 2);
            std::string class_name = classNames[class_id - 1];
            putText(frame, class_name + " " + std::to_string(int(confidence * 100)) + "%", cv::Point(bboxX, bboxY - 10), cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 255, 0), 2);
        }
    }
    auto total_end = std::chrono::steady_clock::now();

    total_fps = 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();
    std::ostringstream stats_ss;
    stats_ss << std::fixed << std::setprecision(2);
    stats_ss << "Inference FPS: " << inference_fps << ", Total FPS: " << total_fps;
    auto stats = stats_ss.str();
    cv::putText(frame, (stats), cv::Point(50, 50), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 255, 0), 2, false);

    cv::imshow("frame", frame);
    if ((char)cv::waitKey(1) == 27)
    {
        std::cout << "ESC pressed" << std::endl;
        cap.release();
        cv::destroyAllWindows();

        reset(true);
    }

    return boundingBox;
}

void opencv_dnn_object_detector_ros::infiniteDarknet()
{
    while (1)
    {
        cap >> frame;
        std::vector<cv::Rect> boundingBoxOutputs = detectDarknet(frame);
        for (size_t i = 0; i < boundingBoxOutputs.size(); i++)
        {
            std::cout << i << " : " << boundingBoxOutputs.at(i) << std::endl;
        }
        std::cout << "****" << std::endl;
    }
}
void opencv_dnn_object_detector_ros::infiniteTensorflow()
{
    while (1)
    {
        cap >> frame;
        std::vector<cv::Rect> boundingBoxOutputs = detectTensorflow(frame);
    }
}
void opencv_dnn_object_detector_ros::cleanup()
{
    std::cout << "Cleanup is called." << std::endl;
    stopDetection = true;
    if (cap.isOpened())
    {
        cap.release();
        cv::destroyAllWindows();
    }
}

void opencv_dnn_object_detector_ros::reset(bool open)
{
    std::cout << "Reset is called." << std::endl;
    cleanup();
    stopDetection = false;

    if (open)
    {
        initialization();
    }
}

bool opencv_dnn_object_detector_ros::openVideoCaptureStream()
{
    try
    {
        std::cout << "openVideoCaptureStream is called." << std::endl;
        cap.open(0);
        if (formatName == "darknet")
        {
            infiniteDarknet();
            return cap.isOpened();
        }
        else if (formatName == "tensorflow")
        {
            infiniteTensorflow();
            return cap.isOpened();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
        return false;
    }
}

void opencv_dnn_object_detector_ros::setCap(cv::VideoCapture &cap)
{
    this->cap = cap;
}
cv::VideoCapture opencv_dnn_object_detector_ros::getCap()
{
    return cap;
}
int main(int argc, char **argv)
{

    ros::init(argc, argv, "opencv_dnn_object_detector_ros_node");

    opencv_dnn_object_detector_ros yolov3;

    ros::spin();

    return 0;
}