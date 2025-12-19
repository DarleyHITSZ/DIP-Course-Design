#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include <ros/package.h>
#include <iostream>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

// 数字模板匹配函数，返回匹配到的数字和中心坐标
pair<int, Point> detectNumber(Mat &frame, vector<Mat> &templates, double threshold = 0.7) {
    Mat gray, blurred, thresh, morph;
    
    // 预处理：转为灰度图
    cvtColor(frame, gray, COLOR_BGR2GRAY);
    
    // 模糊处理以减少噪声
    GaussianBlur(gray, blurred, Size(5, 5), 0);
    
    // 二值化
    cv::threshold(blurred, thresh, 100, 255, THRESH_BINARY_INV | THRESH_OTSU);
    
    // 形态学处理：先腐蚀后膨胀，去除噪点并增强轮廓
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    erode(thresh, morph, kernel, Point(-1, -1), 1);  // 腐蚀
    dilate(morph, morph, kernel, Point(-1, -1), 1);  // 膨胀
    
    // 显示二值化处理后的图像（作为输出参数返回）
    frame = morph.clone();
    
    int bestMatch = -1;
    double bestValue = 0;
    Point bestLoc;
    int bestWidth = 0, bestHeight = 0;
    
    // 对每个模板进行匹配
    for (int i = 0; i < templates.size(); i++) {
        Mat result;
        Mat temp = templates[i];
        
        // 确保模板与待检测图像尺寸匹配
        if (temp.rows > morph.rows || temp.cols > morph.cols)
            continue;
        
        // 模板匹配
        matchTemplate(morph, temp, result, TM_CCOEFF_NORMED);
        
        // 寻找最佳匹配位置
        double minVal, maxVal;
        Point minLoc, maxLoc;
        minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
        
        // 记录最佳匹配
        if (maxVal > threshold && maxVal > bestValue) {
            bestValue = maxVal;
            bestMatch = i;  
            bestLoc = maxLoc;
            bestWidth = temp.cols;
            bestHeight = temp.rows;
        }
    }
    
    // 计算中心位置
    if (bestMatch != -1) {
        Point center;
        center.x = bestLoc.x + bestWidth / 2;
        center.y = bestLoc.y + bestHeight / 2;
        return {bestMatch, center};
    }
    
    return {-1, Point(-1, -1)};  // 未检测到数字
}

int main() {
    // 加载数字模板（假设模板文件名为"template1.png", "template2.png", "template3.png"）
    // 模板图片的完整路径（data文件夹下）
    string template_dir = "/home/eaibot/robot_ws/src/track/data/";
    
    vector<Mat> templates;
    for (int i = 1; i <= 3; i++) {
        string filename = template_dir + "template" + to_string(i) + ".png";
        Mat temp = imread(filename, IMREAD_GRAYSCALE);
        if (temp.empty()) {
            cerr << "无法加载模板文件: " << filename << endl;
            return -1;
        }
        // 对模板进行二值化处理，与检测图像保持一致
        cv::threshold(temp, temp, 100, 255, THRESH_BINARY_INV | THRESH_OTSU);
        
        // 对模板也进行形态学处理，保持与待检测图像处理方式一致
        Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
        erode(temp, temp, kernel, Point(-1, -1), 1);
        dilate(temp, temp, kernel, Point(-1, -1), 1);
        
        templates.push_back(temp);
    }
    
    // 打开摄像头
    VideoCapture cap(0);  // 0表示默认摄像头
    if (!cap.isOpened()) {
        cerr << "无法打开摄像头" << endl;
        return -1;
    }
    
    cout << "数字识别程序启动，按ESC键退出" << endl;
    cout << "检测结果格式: 数字, 中心X坐标, 中心Y坐标" << endl;
    
    // 创建两个窗口，一个显示原始图像及结果，一个显示二值化处理后的图像
    namedWindow("数字识别", WINDOW_AUTOSIZE);
    namedWindow("二值化图像", WINDOW_AUTOSIZE);
    
    while (true) {
        Mat frame, processed;
        cap >> frame;  // 读取一帧图像
        if (frame.empty()) {
            cerr << "无法获取图像帧" << endl;
            break;
        }
        
        // 保存原始图像用于显示
        Mat original = frame.clone();
        
        // 检测数字，同时获取处理后的二值化图像
        pair<int, Point> detection = detectNumber(frame, templates);
        int number = detection.first;
        Point center = detection.second;
        processed = frame;  // 这里的frame已经被处理为二值化图像
        
        // 输出检测结果
        if (number != -1) {
            cout << number << ", " << center.x << ", " << center.y << endl;
            
            // 在原始图像上绘制结果（可视化）
            rectangle(original, Rect(center.x - 10, center.y - 10, 20, 20), Scalar(0, 255, 0), 2);
            putText(original, to_string(number), Point(center.x + 15, center.y - 15), 
                    FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255), 2);
        }
        
        // 显示图像
        imshow("数字识别", original);
        imshow("二值化图像", processed);
        
        // 按ESC键退出
        char c = (char)waitKey(1);
        if (c == 27)
            break;
    }
    
    // 释放资源
    cap.release();
    destroyAllWindows();
    return 0;
}

