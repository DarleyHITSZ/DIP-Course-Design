#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include <ros/package.h>
#include <iostream>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

// 定义结构体，用于返回多组检测结果
struct DetectionResult {
    int number;       // 匹配到的数字
    Point center;     // 数字中心坐标
    double width;     // 数字宽度（对应原始图像尺度）
    double height;    // 数字高度（对应原始图像尺度）
};

// 生成模板金字塔（多尺度模板）
vector<vector<Mat>> generateTemplatePyramid(const vector<Mat>& baseTemplates, const vector<double>& scales) {
    vector<vector<Mat>> pyramid;
    for (const auto& temp : baseTemplates) {
        vector<Mat> scaledTemplates;
        for (double scale : scales) {
            Mat scaledTemp;
            resize(temp, scaledTemp, Size(), scale, scale, INTER_LINEAR);
            scaledTemplates.push_back(scaledTemp);
        }
        pyramid.push_back(scaledTemplates);
    }
    return pyramid;
}

// 模板金字塔匹配函数
DetectionResult detectNumber(Mat &frame, const vector<vector<Mat>>& templatePyramid, 
                           const vector<double>& scales, double threshold = 0.7) {
    Mat gray, blurred, thresh, morph;
    DetectionResult result;
    result.number = -1;
    result.center = Point(-1, -1);
    result.width = 0;
    result.height = 0;
    
    // 预处理：转为灰度图
    cvtColor(frame, gray, COLOR_BGR2GRAY);
    
    // 模糊处理以减少噪声
    GaussianBlur(gray, blurred, Size(5, 5), 0);
    
    // 二值化处理
    cv::threshold(blurred, thresh, 100, 255, THRESH_BINARY_INV | THRESH_OTSU);
    
    // 形态学处理：先腐蚀后膨胀
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    erode(thresh, morph, kernel, Point(-1, -1), 1);
    dilate(morph, morph, kernel, Point(-1, -1), 1);
    
    // 输出预处理后的图像
    frame = morph.clone();
    
    double bestValue = 0;
    Point bestLoc;
    int bestMatch = -1;
    int bestScaleIdx = -1;

    // 使用模板金字塔进行匹配（只缩放模板，不缩放原图）
    for (int i = 0; i < templatePyramid.size(); i++) {
        const auto& scaledTemplates = templatePyramid[i];
        
        for (int s = 0; s < scaledTemplates.size(); s++) {
            const Mat& temp = scaledTemplates[s];
            // 跳过模板尺寸大于原图的情况
            if (temp.rows > morph.rows || temp.cols > morph.cols)
                continue;

            Mat matchResult;
            matchTemplate(morph, temp, matchResult, TM_CCOEFF_NORMED);
            
            double minVal, maxVal;
            Point minLoc, maxLoc;
            minMaxLoc(matchResult, &minVal, &maxVal, &minLoc, &maxLoc);
            
            // 更新最佳匹配结果
            if (maxVal > threshold && maxVal > bestValue) {
                bestValue = maxVal;
                bestMatch = i;
                bestLoc = maxLoc;
                bestScaleIdx = s;
            }
        }
    }
    
    // 计算中心位置并赋值给返回结构体
    if (bestMatch != -1 && bestScaleIdx != -1) {
        double scale = scales[bestScaleIdx];
        const Mat& bestTemp = templatePyramid[bestMatch][bestScaleIdx];
        
        result.number = bestMatch;
        // 计算原始图像中的坐标（不进行缩放逆操作，因为原图未被缩放）
        result.center.x = bestLoc.x + bestTemp.cols / 2;
        result.center.y = bestLoc.y + bestTemp.rows / 2;
        // 计算原始尺寸（基于模板原始尺寸和缩放比例）
        result.width = bestTemp.cols / scale;
        result.height = bestTemp.rows / scale;
    }
    
    return result;
}

int main() {
    // 加载数字模板
    string template_dir = "/home/zdh/robot_ws/src/track/data/";
    
    vector<Mat> baseTemplates;
    for (int i = 1; i <= 3; i++) {
        string filename = template_dir + "template" + to_string(i) + ".png";
        Mat temp = imread(filename, IMREAD_GRAYSCALE);
        if (temp.empty()) {
            cerr << "错误：无法加载模板文件: " << filename << endl;
            return -1;
        }
        // 模板预处理
        cv::threshold(temp, temp, 100, 255, THRESH_BINARY_INV | THRESH_OTSU);
        
        Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
        erode(temp, temp, kernel, Point(-1, -1), 1);
        dilate(temp, temp, kernel, Point(-1, -1), 1);
        
        baseTemplates.push_back(temp);
    }
    
    // 定义缩放比例（模板将按这些比例缩放）
    vector<double> scales = {0.2,0.3,0.4,0.5,0.6,0.8, 1.0};
    
    // 生成模板金字塔
    vector<vector<Mat>> templatePyramid = generateTemplatePyramid(baseTemplates, scales);
    
    // 打开摄像头
    VideoCapture capture;
    capture.open(0);
    if (!capture.isOpened())
    {
      printf("电脑摄像头没有正常打开\n");
      return 0;
    }
    waitKey(1000);
    namedWindow("数字识别", WINDOW_AUTOSIZE);
    
    while (true) {
        Mat frame;
        capture.read(frame);
        if (frame.empty()) {
            cerr << "错误：无法获取摄像头图像帧" << endl;
            break;
        }
        
        Mat original = frame.clone();
        
        // 检测数字（使用模板金字塔）
        DetectionResult detection = detectNumber(frame, templatePyramid, scales);
        
        // 输出并绘制检测结果
        if (detection.number != -1) {
            cout << detection.number << ", " << detection.center.x << ", " << detection.center.y << endl;
            
            // 绘制中心点（红色圆点）
            circle(original, detection.center, 5, Scalar(0, 0, 255), -1);
            
            // 绘制数字标签
            putText(original, to_string(detection.number), 
                    Point(detection.center.x + 15, detection.center.y - 15), 
                    FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
        }
        
        imshow("数字识别", original);
        waitKey(5);
    }
    return 0;
}

