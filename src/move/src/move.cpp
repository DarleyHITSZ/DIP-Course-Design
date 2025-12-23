#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include <ros/package.h>
#include <geometry_msgs/Twist.h>
#include <nav_msgs/Odometry.h>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>

using namespace cv;
using namespace std;

// 定义结构体，用于返回多组检测结果
struct DetectionResult {
    int number;       // 匹配到的数字
    Point center;     // 数字中心坐标
    double width;     // 数字宽度（对应原始图像尺度）
    double height;    // 数字高度（对应原始图像尺度）
    double ratio;     // 数字比例（相对于参考尺寸）
};

// 运动状态机枚举
enum MotionState {
    IDLE = 0,
    ROTATE_TO_TARGET,
    MOVE_FORWARD,
    STOP
};

// 全局变量
MotionState current_state = IDLE;
ros::Publisher vel_pub;
double target_x = -1;          // 目标x坐标
double initial_ratio = 1.0;    // 初始识别到的数字比例
double current_x = 0.0;        // 当前x坐标（里程计）
double start_x = 0.0;          // 开始移动时的x坐标
double forward_distance = 0.0; // 需要前进的距离
double IMAGE_CENTER_X = 320.0; // 图像中心x坐标（假设640x480分辨率）
const double ANGULAR_SPEED = 0.1;    // 旋转角速度
const double LINEAR_SPEED = 0.1;     // 前进速度
const double ROTATION_THRESHOLD = 20.0; // 旋转对准阈值（像素）
const double DISTANCE_THRESHOLD = 0.05; // 距离控制阈值（米）
const double REFERENCE_RATIO = 0.8;  // 参考比例（用于计算前进距离）

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
    result.ratio = 0;
    
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
        
        result.number = bestMatch;  // 数字是1-3
        // 计算原始图像中的坐标
        result.center.x = bestLoc.x + bestTemp.cols / 2;
        result.center.y = bestLoc.y + bestTemp.rows / 2;
        // 计算原始尺寸
        result.width = bestTemp.cols / scale;
        result.height = bestTemp.rows / scale;
        // 计算比例（相对于图像宽度）
        result.ratio = scale;
    }
    
    return result;
}

// 里程计回调函数
void odomCallback(const nav_msgs::OdometryConstPtr& msg) {
    current_x = msg->pose.pose.position.x;
}

// 计算距离
double calcDistance(double x1, double x2) {
    return fabs(x2 - x1);
}

// 运动控制函数
void motionControl() {
    geometry_msgs::Twist vel_msg;
    vel_msg.linear.y = 0.0;
    
    switch (current_state) {
        case IDLE:
            // 空闲状态，不发送速度指令
            vel_msg.linear.x = 0.0;
            vel_msg.angular.z = 0.0;
            break;
            
        case ROTATE_TO_TARGET: {
            // 计算x方向偏差
            double error = target_x - IMAGE_CENTER_X;
            
            if (fabs(error) < ROTATION_THRESHOLD) {
                // 旋转到位，停止旋转
                vel_msg.angular.z = 0.0;
                ROS_INFO("旋转对准完成");
                
                // 判断是否需要前进
                if (initial_ratio < 0.6) {
                    // 计算需要前进的距离，比例越小需要前进越远
                    start_x = current_x;
                    current_state = MOVE_FORWARD;
                    ROS_INFO("准备前进");
                } else {
                    current_state = STOP;
                    ROS_INFO("已对准目标，无需前进");
                }
            } else {
                // 旋转方向：误差为正向右转，误差为负向左转
                vel_msg.angular.z = -ANGULAR_SPEED * (error > 0 ? 1 : -1);
                ROS_INFO_THROTTLE(0.5, "旋转对准中，误差: %.2f 像素", error);
            }
            vel_msg.linear.x = 0.0;
            break;
        }
        
        case MOVE_FORWARD: {
            double distance_moved = calcDistance(start_x, current_x);
            
            if (initial_ratio >= 0.6) {
                // 前进到位
                vel_msg.linear.x = 0.0;
                current_state = STOP;
                ROS_INFO("前进完成");
            } else {
                // 继续前进
                vel_msg.linear.x = LINEAR_SPEED;
                ROS_INFO("前进中");
            }
            vel_msg.angular.z = 0.0;
            break;
        }
        
        case STOP:
            // 停止状态，重置状态机以便下次跟踪
            vel_msg.linear.x = 0.0;
            vel_msg.angular.z = 0.0;
            ROS_INFO_THROTTLE(1, "等待新目标...");
            current_state = IDLE;
            break;
    }
    
    vel_pub.publish(vel_msg);
}

int main(int argc, char**argv) {
    // 初始化ROS节点
    ros::init(argc, argv, "number_tracker");
    ros::NodeHandle nh;
    
    // 创建速度发布者和里程计订阅者
    vel_pub = nh.advertise<geometry_msgs::Twist>("/cmd_vel", 10);
    ros::Subscriber odom_sub = nh.subscribe("/odom", 10, odomCallback);
    
    // 加载数字模板
    string template_dir = "/home/eaibot/robot_ws/src/move/data/";
    
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
    vector<double> scales = {0.2, 0.3, 0.4, 0.5, 0.6,0.8, 1.0};
    
    // 生成模板金字塔
    vector<vector<Mat>> templatePyramid = generateTemplatePyramid(baseTemplates, scales);
    
    // 打开摄像头
    VideoCapture capture;
    capture.open(2);
    if (!capture.isOpened()) {
        printf("摄像头没有正常打开\n");
        return 0;
    }
    waitKey(1000);
    Mat first_frame;
    capture.read(first_frame);
    if (first_frame.empty()) {
	cerr << "无法获取图像帧，无法确定画面尺寸" << endl;
	return -1;
    }
    first_frame = first_frame(Rect(0, 0, first_frame.cols/2, first_frame.rows));
    // 动态计算图像中心X坐标
    double IMAGE_CENTER_X = first_frame.cols / 2.0;  // cols是图像宽度
    ROS_INFO("摄像头画面宽度: %d, 中心X坐标: %.1f", first_frame.cols, IMAGE_CENTER_X);
    namedWindow("数字识别", WINDOW_AUTOSIZE);
    
    // 设置循环频率
    ros::Rate loop_rate(10);
    
    while (ros::ok()) {
        Mat frame;
        capture.read(frame);
        if (frame.empty()) {
            cerr << "错误：无法获取摄像头图像帧" << endl;
            break;
        }
        frame = frame(Rect(0, 0, frame.cols/2, frame.rows));
        Mat original = frame.clone();
        
        // 检测数字（使用模板金字塔）
        DetectionResult detection = detectNumber(frame, templatePyramid, scales);
        
        // 输出并绘制检测结果
        if (detection.number != -1) {
            cout << "数字: " << detection.number << ", 中心坐标: (" << detection.center.x << ", " 
                 << detection.center.y << "), 比例: " << detection.ratio << endl;
            
            // 绘制中心点（红色圆点）
            circle(original, detection.center, 5, Scalar(0, 0, 255), -1);
            
            // 绘制数字标签
            putText(original, to_string(detection.number), 
                    Point(detection.center.x + 15, detection.center.y - 15), 
                    FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
            
            // 绘制图像中心参考线
            line(original, Point(IMAGE_CENTER_X, 0), Point(IMAGE_CENTER_X, original.rows), 
                 Scalar(255, 0, 0), 2);
            
            // 更新目标并跟踪
             target_x = detection.center.x;
             initial_ratio = detection.ratio;
             if(current_state != MOVE_FORWARD) 
                  current_state = ROTATE_TO_TARGET;
            
        }
        
        // 运动控制
        motionControl();
        
        imshow("数字识别", original);
        waitKey(5);
        
        // 处理ROS回调
        ros::spinOnce();
        loop_rate.sleep();
    }
    
    destroyAllWindows();
    capture.release();
    return 0;
}

