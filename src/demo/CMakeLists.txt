cmake_minimum_required(VERSION 3.0.2)
project(demo) # project 名称即为功能包名称

add_compile_options(-std=c++11)

# find_package 查找一些依赖包
find_package(catkin REQUIRED COMPONENTS
roscpp
std_msgs
geometry_msgs
sensor_msgs
cv_bridge
)
find_package(OpenCV REQUIRED)
catkin_package()
#添加头文件路径
include_directories(
${catkin_INCLUDE_DIRS}
${OpenCV_INCLUDE_DIRS}
)
#添加执行文件，即对应节点名称
add_executable(demo src/demo.cpp)

# 添加执行文件需要依赖的链接库
target_link_libraries(demo
${catkin_LIBRARIES}
${OpenCV_LIBRARIES}
)
