#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Odometry.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <geometry_msgs/Twist.h>
#include <vector>
#include <cmath>
#include <string>

using namespace cv;
using namespace std;

// ====================== 配置参数区 ======================
// 1. 摄像头配置
enum CameraType { COMPUTER_CAM, ZED_CAM, REALSENSE_CAM };
const CameraType CAMERA_TYPE = ZED_CAM;       // 选择摄像头
const int COMPUTER_CAM_ID = 0;
const int ZED_CAM_ID = 2;                    
const string REALSENSE_TOPIC = "/camera/color/image_raw";

// 2. 锥桶识别参数
const int H_LOW = 0;
const int H_HIGH = 180;
const int S_LOW = 150;
const int S_HIGH = 255;
const int V_LOW = 35;
const int V_HIGH = 255;
const int MIN_CONTOUR_AREA = 200; 

// 3. 锥桶路径运动控制参数
const double D1 = 1.4;
const double D2 = 0.8;
const double D3 = 1.2;
const double ANG1 = 36;
const double ANG2 = 36;
const double ANG3 = 55;
const double CONE_LINEAR_SPEED = 0.2;    // 锥桶阶段直行速度
const double CONE_ANGULAR_SPEED = 0.5;   // 锥桶阶段旋转角速度

// 4. 数字识别与跟踪参数
const string TEMPLATE_DIR = "/home/eaibot/robot_ws/src/demo/data/";  // 数字模板路径
const vector<double> SCALES = {0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0};  // 模板缩放比例
const double MATCH_THRESHOLD = 0.7;      // 模板匹配阈值
const double NUM_ANGULAR_SPEED = 0.1;    // 数字跟踪旋转角速度
const double NUM_LINEAR_SPEED = 0.1;     // 数字跟踪前进速度
const double ROTATION_THRESHOLD = 20.0;  // 旋转对准阈值（像素）
const double DISTANCE_THRESHOLD = 0.05;  // 距离控制阈值（米）
const double REFERENCE_RATIO = 0.8;      // 参考比例（用于计算前进距离）

// ====================== 全局变量与结构体 ======================
// 数字检测结果结构体
struct DetectionResult {
    int number;       // 匹配到的数字
    Point center;     // 数字中心坐标
    double width;     // 数字宽度（对应原始图像尺度）
    double height;    // 数字高度（对应原始图像尺度）
    double ratio;     // 数字比例（相对于参考尺寸）
};

// 合并后的状态机枚举
enum MotionState {
    IDLE = 0,
    // 锥桶跟踪阶段
    GO_STRAIGHT_D1,
    TURN_RIGHT_ANG1,
    GO_STRAIGHT_D2,
    TURN_RIGHT_ANG2,
    GO_STRAIGHT_D3,
    TURN_RIGHT_ANG3,
    GO_STRAIGHT_FINAL,
    // 数字跟踪阶段
    ROTATE_TO_NUMBER,
    MOVE_TO_NUMBER,
    STOP
};

// 全局变量
MotionState current_state = IDLE;
bool cone_detected = false;
bool mission_started = false;
double current_x = 0.0, current_y = 0.0;
double start_x = 0.0, start_y = 0.0;
double current_yaw = 0.0;
double start_yaw = 0.0;
double IMAGE_CENTER_X = 320.0;  // 图像中心x坐标
double target_x = -1;           // 数字目标x坐标
double initial_ratio = 1.0;     // 初始识别到的数字比例
ros::Publisher vel_pub;         // 速度发布者
VideoCapture capture;
Mat frame_msg;
vector<vector<Mat>> templatePyramid;  // 数字模板金字塔

// ====================== 锥桶识别与控制函数 ======================
// Realsense图像回调
void realsenseImgCallback(const sensor_msgs::Image::ConstPtr& msg) {
    try {
        frame_msg = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8)->image;
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("Realsense图像转换失败: %s", e.what());
    }
}

// 锥桶识别
void detectCone(const Mat& img, bool& detected) {
    Mat hsv_img, mask;
    cvtColor(img, hsv_img, COLOR_BGR2HSV);
    inRange(hsv_img, Scalar(H_LOW, S_LOW, V_LOW), Scalar(H_HIGH, S_HIGH, V_HIGH), mask);
    
    // 形态学去噪
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    morphologyEx(mask, mask, MORPH_OPEN, kernel);
    morphologyEx(mask, mask, MORPH_CLOSE, kernel);
    
    // 轮廓检测
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(mask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    detected = false;
    for (size_t i = 0; i < contours.size(); ++i) {
        if (contourArea(contours[i]) > MIN_CONTOUR_AREA) {
            detected = true;
            break;
        }
    }
    imshow("锥桶掩码", mask);
}

// ====================== 数字识别与控制函数 ======================
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

// 加载数字模板
bool loadNumberTemplates(vector<Mat>& baseTemplates, const string& dir) {
    for (int i = 1; i <= 3; i++) {
        string filename = dir + "template" + to_string(i) + ".png";
        Mat temp = imread(filename, IMREAD_GRAYSCALE);
        if (temp.empty()) {
            cerr << "错误：无法加载模板文件: " << filename << endl;
            return false;
        }
        // 模板预处理
        cv::threshold(temp, temp, 100, 255, THRESH_BINARY_INV | THRESH_OTSU);
        
        Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
        erode(temp, temp, kernel, Point(-1, -1), 1);
        dilate(temp, temp, kernel, Point(-1, -1), 1);
        
        baseTemplates.push_back(temp);
    }
    return true;
}

// 模板金字塔匹配函数
DetectionResult detectNumber(Mat &frame) {
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
    
    double bestValue = 0;
    Point bestLoc;
    int bestMatch = -1;
    int bestScaleIdx = -1;

    // 使用模板金字塔进行匹配
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
            if (maxVal > MATCH_THRESHOLD && maxVal > bestValue) {
                bestValue = maxVal;
                bestMatch = i;
                bestLoc = maxLoc;
                bestScaleIdx = s;
            }
        }
    }
    
    // 计算中心位置并赋值给返回结构体
    if (bestMatch != -1 && bestScaleIdx != -1) {
        double scale = SCALES[bestScaleIdx];
        const Mat& bestTemp = templatePyramid[bestMatch][bestScaleIdx];
        
        result.number = bestMatch;  // 数字是0-2
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

// ====================== 通用工具函数 ======================
// 距离计算
double calcDistance(double x1, double y1, double x2, double y2) {
    return sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));
}

// 转弯角度计算
double calcTurnAngle(double start_yaw, double current_yaw) {
    double diff = current_yaw - start_yaw;
    if (diff > M_PI) diff -= 2 * M_PI;
    if (diff < -M_PI) diff += 2 * M_PI;
    return fabs(diff);
}

// 里程计回调
void odomCallback(const nav_msgs::OdometryConstPtr& msg) {
    current_x = msg->pose.pose.position.x;
    current_y = msg->pose.pose.position.y;
}

// IMU回调
void imuCallback(const sensor_msgs::ImuConstPtr& msg) {
    double qx = msg->orientation.x;
    double qy = msg->orientation.y;
    double qz = msg->orientation.z;
    double qw = msg->orientation.w;
    current_yaw = atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz));
}

// ====================== 运动控制函数 ======================
void motionControl(bool cone_detected, const DetectionResult& number_detected) {
    geometry_msgs::Twist vel_msg;
    vel_msg.linear.y = 0.0;
    
    switch (current_state) {
        // 锥桶跟踪阶段
        case GO_STRAIGHT_D1: {
            double distance = calcDistance(start_x, start_y, current_x, current_y);
            if (distance >= D1) {
                current_state = TURN_RIGHT_ANG1;
                start_yaw = current_yaw;
                ROS_INFO("完成D1直行（%.2fm），开始右转ANG1=%.1f°", distance, ANG1);
                vel_msg.linear.x = 0.0;
                vel_msg.angular.z = -CONE_ANGULAR_SPEED; 
            } else {
                vel_msg.linear.x = CONE_LINEAR_SPEED;
                vel_msg.angular.z = 0.0;
                ROS_INFO("直行D1：已行驶%.2fm/%.2fm", distance, D1);
            }
            break;
        }
        
        case TURN_RIGHT_ANG1: {
            double target_angle = ANG1 * M_PI / 180.0;
            double turned_angle = calcTurnAngle(start_yaw, current_yaw);
            if (turned_angle >= target_angle) {
                current_state = GO_STRAIGHT_D2;
                start_x = current_x;
                start_y = current_y;
                ROS_INFO("完成ANG1右转（%.1f°），开始直行D2=%.2fm", ANG1, D2);
                vel_msg.angular.z = 0.0;
                vel_msg.linear.x = CONE_LINEAR_SPEED;
            } else {
                vel_msg.linear.x = 0.0;
                vel_msg.angular.z = -CONE_ANGULAR_SPEED;  
                ROS_INFO("右转ANG1：已转%.1f°/%.1f°", turned_angle * 180 / M_PI, ANG1);
            }
            break;
        }
        
        case GO_STRAIGHT_D2: {
            double distance = calcDistance(start_x, start_y, current_x, current_y);
            if (distance >= D2) {
                current_state = TURN_RIGHT_ANG2;
                start_yaw = current_yaw;
                ROS_INFO("完成D2直行（%.2fm），开始转ANG2=%.1f°", distance, ANG2);
                vel_msg.linear.x = 0.0;
                vel_msg.angular.z = CONE_ANGULAR_SPEED;  
            } else {
                vel_msg.linear.x = CONE_LINEAR_SPEED;
                vel_msg.angular.z = 0.0;
                ROS_INFO("直行D2：已行驶%.2fm/%.2fm", distance, D2);
            }
            break;
        }
        
        case TURN_RIGHT_ANG2: {
            double target_angle = ANG2 * M_PI / 180.0;
            double turned_angle = calcTurnAngle(start_yaw, current_yaw);
            if (turned_angle >= target_angle) {
                current_state = GO_STRAIGHT_D3;
                start_x = current_x;
                start_y = current_y;
                ROS_INFO("完成ANG2转（%.1f°），直行", ANG2);
                vel_msg.angular.z = 0.0;
                vel_msg.linear.x = CONE_LINEAR_SPEED;
            } else {
                vel_msg.linear.x = 0.0;
                vel_msg.angular.z = CONE_ANGULAR_SPEED; 
                ROS_INFO("转ANG2：已转%.1f°/%.1f°", turned_angle * 180 / M_PI, ANG2);
            }
            break;
        }
        
        case GO_STRAIGHT_D3: {
            double distance = calcDistance(start_x, start_y, current_x, current_y);
            if (distance >= D3) {
                current_state = TURN_RIGHT_ANG3;
                start_yaw = current_yaw;
                ROS_INFO("完成D3直行（%.2fm），开始右转ANG3=%.1f°", distance, ANG3);
                vel_msg.linear.x = 0.0;
                vel_msg.angular.z = -CONE_ANGULAR_SPEED;  
            } else {
                vel_msg.linear.x = CONE_LINEAR_SPEED;
                vel_msg.angular.z = 0.0;
                ROS_INFO("直行D3：已行驶%.2fm/%.2fm", distance, D3);
            }
            break;
        }
        
        case TURN_RIGHT_ANG3: {
            double target_angle = ANG3 * M_PI / 180.0;
            double turned_angle = calcTurnAngle(start_yaw, current_yaw);
            if (turned_angle >= target_angle) {
                current_state = GO_STRAIGHT_FINAL;
                ROS_INFO("完成ANG3转（%.1f°），直行", ANG3);
                vel_msg.angular.z = 0.0;
                vel_msg.linear.x = CONE_LINEAR_SPEED;
            } else {
                vel_msg.linear.x = 0.0;
                vel_msg.angular.z = -CONE_ANGULAR_SPEED; 
                ROS_INFO("转ANG3：已转%.1f°/%.1f°", turned_angle * 180 / M_PI, ANG3);
            }
            break;
        }
        
        case GO_STRAIGHT_FINAL:
            // 直行一段时间后进入数字跟踪阶段
            vel_msg.linear.x = CONE_LINEAR_SPEED;
            vel_msg.angular.z = 0.0;
            ROS_INFO_THROTTLE(1, "持续直行中，即将进入数字跟踪阶段");
            
            // 这里可以根据实际需求设置进入数字跟踪的条件

            if (cone_detected == false) {  
                current_state = STOP;
                ROS_INFO("已驶出道路，开始数字跟踪");
                vel_msg.linear.x = 0.0;
            }
            break;
        
        // 数字跟踪阶段
        case ROTATE_TO_NUMBER: {
            // 计算x方向偏差
            double error = number_detected.center.x - IMAGE_CENTER_X;
            target_x = number_detected.center.x;
            initial_ratio = number_detected.ratio;
            
            if (fabs(error) < ROTATION_THRESHOLD) {
                // 旋转到位，停止旋转
                vel_msg.angular.z = 0.0;
                ROS_INFO("旋转对准数字完成");
                
                // 判断是否需要前进
                if (initial_ratio < 0.6) {
                    current_state = MOVE_TO_NUMBER;
                    ROS_INFO("准备向数字前进");
                } else {
                    current_state = STOP;
                    ROS_INFO("已对准数字，无需前进");
                }
            } else {
                // 旋转方向：误差为正向右转，误差为负向左转
                vel_msg.angular.z = -NUM_ANGULAR_SPEED * (error > 0 ? 1 : -1);
                ROS_INFO_THROTTLE(0.5, "旋转对准数字中，误差: %.2f 像素", error);
            }
            vel_msg.linear.x = 0.0;
            break;
        }
        
        case MOVE_TO_NUMBER: {    
            if (number_detected.ratio >= 0.6) {
                // 前进到位
                vel_msg.linear.x = 0.0;
                current_state = STOP;
                ROS_INFO("已到达数字目标位置");
            } else {
                // 继续前进
                vel_msg.linear.x = NUM_LINEAR_SPEED;
                ROS_INFO("向数字前进中");
            }
            vel_msg.angular.z = 0.0;
            break;
        }
        
        case IDLE:
            // 等待锥桶出现
            vel_msg.linear.x = 0.0;
            vel_msg.angular.z = 0.0;
            if (!mission_started && cone_detected) {
                current_state = GO_STRAIGHT_D1;
                start_x = current_x;
                start_y = current_y;
                mission_started = true;
                ROS_INFO("检测到锥桶，启动任务");
            } 
            break;
            
        case STOP:
            // 任务完成，停止
            vel_msg.linear.x = 0.0;
            vel_msg.angular.z = 0.0;
            ROS_INFO_THROTTLE(1, "停止运动");
            break;
    }
    
    vel_pub.publish(vel_msg);
}

// 摄像头初始化
bool initCamera() {
    if (CAMERA_TYPE == COMPUTER_CAM) {
        capture.open(COMPUTER_CAM_ID);
        if (!capture.isOpened()) {
            ROS_ERROR("电脑摄像头初始化失败！");
            return false;
        }
        ROS_INFO("初始化电脑摄像头成功");
    } else if (CAMERA_TYPE == ZED_CAM) {
        capture.open(ZED_CAM_ID);
        if (!capture.isOpened()) {
                ROS_ERROR("ZED摄像头初始化失败！");
                return false;
        }
        ROS_INFO("初始化ZED摄像头成功");
        waitKey(1000);  
    } else if (CAMERA_TYPE == REALSENSE_CAM) {
        ROS_INFO("初始化Realsense摄像头成功");
    } else {
        ROS_ERROR("未知摄像头类型");
        return false;
    }
    
    // 确定图像中心
    if (CAMERA_TYPE != REALSENSE_CAM) {
        Mat first_frame;
        capture.read(first_frame);
        if (!first_frame.empty()) {
            if (CAMERA_TYPE == ZED_CAM) {
                first_frame = first_frame(Rect(0, 0, first_frame.cols/2, first_frame.rows));
            }
            IMAGE_CENTER_X = first_frame.cols / 2.0;
            ROS_INFO("摄像头画面宽度: %d, 中心X坐标: %.1f", first_frame.cols, IMAGE_CENTER_X);
        }
    }
    
    return true;
}

// 初始化数字模板
bool initNumberTemplates() {
    vector<Mat> baseTemplates;
    if (!loadNumberTemplates(baseTemplates, TEMPLATE_DIR)) {
        return false;
    }
    templatePyramid = generateTemplatePyramid(baseTemplates, SCALES);
    return true;
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "cone_and_number_tracker");
    ros::NodeHandle nh;
    
    // 1. 初始化摄像头和数字模板
    if (!initCamera() || !initNumberTemplates()) {
        return -1;
    }
    
    // 2. 创建订阅者/发布者
    ros::Subscriber odom_sub = nh.subscribe("/odom", 10, odomCallback);
    ros::Subscriber imu_sub = nh.subscribe("/imu", 10, imuCallback);
    ros::Subscriber rs_img_sub;
    if (CAMERA_TYPE == REALSENSE_CAM) {
        rs_img_sub = nh.subscribe(REALSENSE_TOPIC, 10, realsenseImgCallback);
    }
    vel_pub = nh.advertise<geometry_msgs::Twist>("/cmd_vel", 10);
    
    // 3. 创建可视化窗口
    namedWindow("锥桶掩码", WINDOW_NORMAL);
    namedWindow("数字识别", WINDOW_AUTOSIZE);
    
    // 4. 主循环10Hz
    //current_state=STOP;
    ros::Rate loop_rate(10);
    while (ros::ok()) {
        Mat frame;
        bool is_frame_valid = false;
        
        // 获取图像
        if (CAMERA_TYPE == COMPUTER_CAM || CAMERA_TYPE == ZED_CAM) {
            capture.read(frame);
            if (!frame.empty()) {
                is_frame_valid = true;
                if (CAMERA_TYPE == ZED_CAM) {
                    frame = frame(Rect(0, 0, frame.cols/2, frame.rows));  // 截取左目
                }
            } else {
                ROS_WARN_THROTTLE(1, "未获取到摄像头图像");
            }
        } else if (CAMERA_TYPE == REALSENSE_CAM) {
            if (!frame_msg.empty()) {
                frame = frame_msg.clone();
                is_frame_valid = true;
            } else {
                ROS_WARN_THROTTLE(1, "未获取到Realsense图像");
            }
        }
        
        // 处理图像并检测目标
        DetectionResult number_result;
        number_result.number = -1;
        
        if (is_frame_valid) {
            Mat original = frame.clone();
            
            // 锥桶检测（仅在锥桶阶段有效）
            if (current_state < ROTATE_TO_NUMBER) {
                detectCone(original, cone_detected);
            }
            
            // 数字检测（仅在数字跟踪阶段有效）
            else
                if (current_state >= ROTATE_TO_NUMBER) {
                    number_result = detectNumber(original);
                    
                    // 绘制数字检测结果
                    if (number_result.number != -1) {
                        // 绘制中心点（红色圆点）
                        circle(original, number_result.center, 5, Scalar(0, 0, 255), -1);
                        
                        // 绘制数字标签
                        putText(original, to_string(number_result.number), 
                                Point(number_result.center.x + 15, number_result.center.y - 15), 
                                FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
                        
                        // 绘制图像中心参考线
                        line(original, Point(IMAGE_CENTER_X, 0), Point(IMAGE_CENTER_X, original.rows), 
                            Scalar(255, 0, 0), 2);

                        if(current_state != MOVE_TO_NUMBER){
                             current_state =  ROTATE_TO_NUMBER;
                        }
                    }
                    else
                        if(current_state ==  MOVE_TO_NUMBER){
                            current_state =  STOP;
                        }
                    imshow("数字识别", original);
                }
        }
        
        // 运动控制
        motionControl(cone_detected, number_result);
        
        waitKey(5);
        ros::spinOnce();
        loop_rate.sleep();
    }
    
    destroyAllWindows();
    if (CAMERA_TYPE == COMPUTER_CAM || CAMERA_TYPE == ZED_CAM) {
        capture.release();
    }
    return 0;
}
