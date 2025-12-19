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

// ====================== 1. 摄像头配置 ======================
enum CameraType { COMPUTER_CAM, ZED_CAM, REALSENSE_CAM };
const CameraType CAMERA_TYPE = ZED_CAM;       // 选择摄像头
const int COMPUTER_CAM_ID = 0;
const int ZED_CAM_ID = 2;                    
const string REALSENSE_TOPIC = "/camera/color/image_raw";

// ====================== 2. 锥桶识别参数 ======================
const int H_LOW = 0;
const int H_HIGH = 180;
const int S_LOW = 150;
const int S_HIGH = 255;
const int V_LOW = 35;
const int V_HIGH = 255;
const int MIN_CONTOUR_AREA = 200; 

// ====================== 3. 运动控制参数======================
const double D1 = 1.3;
const double D2 = 1.85;
const double ANG1 = 13.0;
const double ANG2 = 23.0;
const double LINEAR_SPEED = 0.2;    // 直行速度
const double ANGULAR_SPEED = 0.5;    // 旋转角速度

// ====================== 4. 固定参数 ======================
bool cone_detected = false;
bool mission_started = false;
double current_x = 0.0, current_y = 0.0;
double start_x = 0.0, start_y = 0.0;
double current_yaw = 0.0;
double start_yaw = 0.0;
double cone_detected_time = 0.0;
VideoCapture capture;
Mat frame_msg;
ros::Publisher vel_pub;  // 全局发布者

// 状态机枚举
enum MotionState {
    IDLE = 0,
    GO_STRAIGHT_D1,
    TURN_RIGHT_ANG1,
    GO_STRAIGHT_D2,
    TURN_RIGHT_ANG2,
    GO_STRAIGHT_FINAL,
    STOP
};
MotionState current_state = IDLE;

// Realsense图像回调
void realsenseImgCallback(const sensor_msgs::Image::ConstPtr& msg) {
    try {
        frame_msg = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8)->image;
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("Realsense图像转换失败: %s", e.what());
    }
}

// 锥桶识别
void detectCone(const Mat& img) {
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
    
    cone_detected = false;
    for (size_t i = 0; i < contours.size(); ++i) {
        if (contourArea(contours[i]) > MIN_CONTOUR_AREA) {
            cone_detected = true;
            break;
        }
    }
    
    imshow("锥桶掩码", mask);
    imshow("原始图像", img);
    waitKey(1);
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

// 运动控制
void motionControl() {
    geometry_msgs::Twist vel_msg;
    ros::Time now = ros::Time::now();
    
    if (!cone_detected) {                 // 锥桶丢失处理
        if (current_state == GO_STRAIGHT_FINAL) {
            current_state = STOP;
            mission_started = false;
            ROS_INFO("锥桶丢失，停止");
        }
        if (current_state == IDLE) {
            ROS_INFO_THROTTLE(1, "未检测到锥桶");
        }
    }
    else
    	if (!mission_started) {               // 检测到锥桶，快速启动任务
            current_state = GO_STRAIGHT_D1;
            start_x = current_x;
            start_y = current_y;
            mission_started = true;
            ROS_INFO("启动任务：开始直行D1=%.2fm，速度=%.2fm/s", D1, LINEAR_SPEED);
        }
    
    // 各状态运动控制
    switch (current_state) {
        case GO_STRAIGHT_D1: {
            double distance = calcDistance(start_x, start_y, current_x, current_y);
            if (distance >= D1) {
                current_state = TURN_RIGHT_ANG1;
                start_yaw = current_yaw;
                ROS_INFO("完成D1直行（%.2fm），开始右转ANG1=%.1f°，角速度=%.2frad/s", 
                         distance, ANG1, ANGULAR_SPEED);
                vel_msg.linear.x = 0.0;
                vel_msg.angular.z = -ANGULAR_SPEED; 
            } else {
                vel_msg.linear.x = LINEAR_SPEED;
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
                vel_msg.linear.x = LINEAR_SPEED;
            } else {
                vel_msg.linear.x = 0.0;
                vel_msg.angular.z = -ANGULAR_SPEED;  
                ROS_INFO("右转ANG1：已转%.1f°/%.1f°", turned_angle * 180 / M_PI, ANG1);
            }
            break;
        }
        
        case GO_STRAIGHT_D2: {
            double distance = calcDistance(start_x, start_y, current_x, current_y);
            if (distance >= D2) {
                current_state = TURN_RIGHT_ANG2;
                start_yaw = current_yaw;
                ROS_INFO("完成D2直行（%.2fm），开始右转ANG2=%.1f°", distance, ANG2);
                vel_msg.linear.x = 0.0;
                vel_msg.angular.z = -ANGULAR_SPEED;  
            } else {
                vel_msg.linear.x = LINEAR_SPEED;
                vel_msg.angular.z = 0.0;
                ROS_INFO("直行D2：已行驶%.2fm/%.2fm", distance, D2);
            }
            break;
        }
        
        case TURN_RIGHT_ANG2: {
            double target_angle = ANG2 * M_PI / 180.0;
            double turned_angle = calcTurnAngle(start_yaw, current_yaw);
            if (turned_angle >= target_angle) {
                current_state = GO_STRAIGHT_FINAL;
                ROS_INFO("完成ANG2右转（%.1f°），持续直行", ANG2);
                vel_msg.angular.z = 0.0;
                vel_msg.linear.x = LINEAR_SPEED;
            } else {
                vel_msg.linear.x = 0.0;
                vel_msg.angular.z = -ANGULAR_SPEED; 
                ROS_INFO("右转ANG2：已转%.1f°/%.1f°", turned_angle * 180 / M_PI, ANG2);
            }
            break;
        }
        
        case GO_STRAIGHT_FINAL:
            vel_msg.linear.x = LINEAR_SPEED;
            vel_msg.angular.z = 0.0;
            ROS_INFO_THROTTLE(1, "持续直行中");
            break;
        
        default:
            vel_msg.linear.x = 0.0;
            vel_msg.angular.z = 0.0;
            break;
    }
    vel_msg.linear.y = 0.0;
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
    return true;
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "cone_follow_task");
    ros::NodeHandle nh;
    
    // 1. 初始化摄像头
    if (!initCamera()) {
        return -1;
    }
    
    // 2. 创建订阅者/发布者
    ros::Subscriber odom_sub = nh.subscribe("/odom", 10, odomCallback);
    ros::Subscriber imu_sub = nh.subscribe("/imu", 10, imuCallback);
    ros::Subscriber rs_img_sub;
    if (CAMERA_TYPE == REALSENSE_CAM) {
        rs_img_sub = nh.subscribe(REALSENSE_TOPIC, 10, realsenseImgCallback);
    }
    // 全局发布者，确保所有函数可访问
    vel_pub = nh.advertise<geometry_msgs::Twist>("/cmd_vel", 10);
    
    // 3. 创建可视化窗口
    namedWindow("原始图像", WINDOW_NORMAL);
    namedWindow("锥桶掩码", WINDOW_NORMAL);
    
    // 4. 主循环10Hz
    ros::Rate loop_rate(10);
    while (ros::ok()) {
        Mat frame;
        bool is_frame_valid = false;
        
        // 图像获取
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
        
        // 锥桶识别
        if (is_frame_valid) {
            detectCone(frame);
        }
        
        // 运动控制
        motionControl();
        
        ros::spinOnce();
        loop_rate.sleep();
    }
    
    destroyAllWindows();
    if (CAMERA_TYPE == COMPUTER_CAM || CAMERA_TYPE == ZED_CAM) {
        capture.release();
    }
    return 0;
}

