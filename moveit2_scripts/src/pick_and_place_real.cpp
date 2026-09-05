#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>

#include <moveit_msgs/msg/display_robot_state.hpp>
#include <moveit_msgs/msg/display_trajectory.hpp>
#include <moveit_msgs/msg/robot_trajectory.hpp>

#include <chrono>
#include <cmath>
#include <memory>
#include <thread>
#include <vector>

static const rclcpp::Logger LOGGER = rclcpp::get_logger("move_group_node");
static const std::string PLANNING_GROUP_ROBOT = "ur_manipulator";
static const std::string PLANNING_GROUP_GRIPPER = "gripper";

class PickAndPlace {
public:
  PickAndPlace(rclcpp::Node::SharedPtr base_node_)
      : base_node_(base_node_) {
    RCLCPP_INFO(LOGGER, "Initializing Class: Pick And Place...");

    rclcpp::NodeOptions node_options;
    node_options.automatically_declare_parameters_from_overrides(true);

    move_group_node_ = rclcpp::Node::make_shared("move_group_node", node_options);
    executor_.add_node(move_group_node_);
    std::thread([this]() { this->executor_.spin(); }).detach();

    move_group_robot_ = std::make_shared<MoveGroupInterface>(
        move_group_node_, PLANNING_GROUP_ROBOT);
    move_group_gripper_ = std::make_shared<MoveGroupInterface>(
        move_group_node_, PLANNING_GROUP_GRIPPER);

    joint_model_group_robot_ = move_group_robot_->getCurrentState()->getJointModelGroup(PLANNING_GROUP_ROBOT);
    joint_model_group_gripper_ = move_group_gripper_->getCurrentState()->getJointModelGroup(PLANNING_GROUP_GRIPPER);

    current_state_robot_ = move_group_robot_->getCurrentState(10);
    current_state_robot_->copyJointGroupPositions(joint_model_group_robot_,
                                                  joint_group_positions_robot_);
    current_state_gripper_ = move_group_gripper_->getCurrentState(10);
    current_state_gripper_->copyJointGroupPositions(joint_model_group_gripper_, joint_group_positions_gripper_);

    move_group_robot_->setStartStateToCurrentState();
    move_group_gripper_->setStartStateToCurrentState();

    // Add the table to prevent crashes
    add_collision_objects();

    RCLCPP_INFO(LOGGER, "Class Initialized: Pick And Place");
  }

  void execute_trajectory_plan() {
    RCLCPP_INFO(LOGGER, "Executing Pick And Place Sequence...");
/*
    // 1. Initial Position (Ready Pose)
    RCLCPP_INFO(LOGGER, "Going to Initial Position...");
    setup_joint_value_target(+0.000, -1.5708, +1.5708, -1.5708, -1.5708, +0.000);
    plan_trajectory_kinematics();
    execute_trajectory_kinematics();
*/

    // 2. Go to the position of the object to grasp 
    RCLCPP_INFO(LOGGER, "Going to Pregrasp Position...");
    setup_goal_pose_target(+0.343, +0.132, +0.220, -1.000, +0.000, +0.000, +0.000);
    plan_trajectory_kinematics();
    execute_trajectory_kinematics();

    // 3. Open the gripper
    RCLCPP_INFO(LOGGER, "Opening Gripper...");
    setup_joint_value_gripper(0.000);
    plan_trajectory_gripper();
    execute_trajectory_gripper();

    // 4. Approach 
    RCLCPP_INFO(LOGGER, "Approaching (Cartesian Line)...");
    move_group_robot_->setMaxVelocityScalingFactor(0.1);
    execute_cartesian_trajectory(+0.343, +0.132, (target_pose_robot_.position.z - delta_), -1.000, +0.000, +0.000, +0.000);

    //std::this_thread::sleep_for(std::chrono::seconds(30));

    // 5. Close the gripper
    RCLCPP_INFO(LOGGER, "Closing Gripper...");
    setup_joint_value_gripper(+0.628); 
    plan_trajectory_gripper();
    execute_trajectory_gripper();

    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    // 6. lift the object up in a start line
    RCLCPP_INFO(LOGGER, "Retreating...");
    move_group_robot_->setMaxVelocityScalingFactor(0.25);
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    execute_cartesian_trajectory(+0.343, +0.132, (target_pose_robot_.position.z - delta_), -1.000, +0.000, +0.000, +0.000);

    // 7. Turn ONLY the shoulder joint 180 degrees 
    RCLCPP_INFO(LOGGER, "Reading current state and turning 180 degrees...");
    
    current_state_robot_ = move_group_robot_->getCurrentState(10);
    current_state_robot_->copyJointGroupPositions(joint_model_group_robot_, joint_group_positions_robot_);
    joint_group_positions_robot_[0] += M_PI; 
    move_group_robot_->setJointValueTarget(joint_group_positions_robot_);
    move_group_robot_->setMaxVelocityScalingFactor(0.1);
    plan_trajectory_kinematics();
    execute_trajectory_kinematics();

    // 8. Open the gripper
    RCLCPP_INFO(LOGGER, "Opening Gripper...");
    move_group_robot_->setMaxVelocityScalingFactor(1.0);
    setup_joint_value_gripper(0.000); 
    plan_trajectory_gripper();
    execute_trajectory_gripper();
    RCLCPP_INFO(LOGGER, "Gripper Opened");

    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    // 9. Return to Home Position
    RCLCPP_INFO(LOGGER, "Going to Home Position...");
    setup_joint_value_target(0.000, -1.5708, 0.000, -1.5708, 0.000, 0.000);
    plan_trajectory_kinematics();
    execute_trajectory_kinematics();

    RCLCPP_INFO(LOGGER, "Pick And Place Trajectory Execution Complete");

    
  }

private:
  using MoveGroupInterface = moveit::planning_interface::MoveGroupInterface;
  using PlanningSceneInterface = moveit::planning_interface::PlanningSceneInterface;
  using JointModelGroup = moveit::core::JointModelGroup;
  using RobotStatePtr = moveit::core::RobotStatePtr;
  using Plan = MoveGroupInterface::Plan;
  using Pose = geometry_msgs::msg::Pose;

  rclcpp::Node::SharedPtr base_node_;
  rclcpp::Node::SharedPtr move_group_node_;
  rclcpp::executors::SingleThreadedExecutor executor_;

  std::shared_ptr<MoveGroupInterface> move_group_robot_;
  std::shared_ptr<MoveGroupInterface> move_group_gripper_;
  PlanningSceneInterface planning_scene_interface_; 

  const JointModelGroup *joint_model_group_robot_;
  const JointModelGroup *joint_model_group_gripper_;

  std::vector<double> joint_group_positions_robot_;
  RobotStatePtr current_state_robot_;
  Plan kinematics_trajectory_plan_;
  Pose target_pose_robot_;
  bool plan_success_robot_ = false;
  std::vector<double> joint_group_positions_gripper_;
  RobotStatePtr current_state_gripper_;
  Plan gripper_trajectory_plan_;
  bool plan_success_gripper_ = false;
  
  float delta_ = 0.020; // meters

  void add_collision_objects() {
    moveit_msgs::msg::CollisionObject collision_object;
    collision_object.header.frame_id = move_group_robot_->getPlanningFrame();
    collision_object.id = "table";

    shape_msgs::msg::SolidPrimitive primitive;
    primitive.type = primitive.BOX;
    primitive.dimensions.resize(3);
    primitive.dimensions[0] = 1.5;
    primitive.dimensions[1] = 1.5;
    primitive.dimensions[2] = 0.05;

    geometry_msgs::msg::Pose box_pose;
    box_pose.orientation.w = 1.0;
    box_pose.position.x = 0.0;
    box_pose.position.y = 0.0;
    box_pose.position.z = -0.03;

    collision_object.primitives.push_back(primitive);
    collision_object.primitive_poses.push_back(box_pose);
    collision_object.operation = collision_object.ADD;

    std::vector<moveit_msgs::msg::CollisionObject> collision_objects;
    collision_objects.push_back(collision_object);

    planning_scene_interface_.applyCollisionObjects(collision_objects);
  }

  void setup_joint_value_target(float angle0, float angle1, float angle2,
                                float angle3, float angle4, float angle5) {
    joint_group_positions_robot_[0] = angle0;
    joint_group_positions_robot_[1] = angle1;
    joint_group_positions_robot_[2] = angle2;
    joint_group_positions_robot_[3] = angle3;
    joint_group_positions_robot_[4] = angle4;
    joint_group_positions_robot_[5] = angle5;
    move_group_robot_->setJointValueTarget(joint_group_positions_robot_);
  }

  void setup_goal_pose_target(float pos_x, float pos_y, float pos_z,
                              float quat_x, float quat_y, float quat_z,
                              float quat_w) {
    target_pose_robot_.position.x = pos_x;
    target_pose_robot_.position.y = pos_y;
    target_pose_robot_.position.z = pos_z;
    target_pose_robot_.orientation.x = quat_x;
    target_pose_robot_.orientation.y = quat_y;
    target_pose_robot_.orientation.z = quat_z;
    target_pose_robot_.orientation.w = quat_w;
    move_group_robot_->setPoseTarget(target_pose_robot_);
  }

  void execute_cartesian_trajectory(float pos_x, float pos_y, float pos_z,
                                    float quat_x, float quat_y, float quat_z,
                                    float quat_w) {
    std::vector<Pose> waypoints;
    Pose target_pose;
    
    target_pose.position.x = pos_x;
    target_pose.position.y = pos_y;
    target_pose.position.z = pos_z;
    target_pose.orientation.x = quat_x;
    target_pose.orientation.y = quat_y;
    target_pose.orientation.z = quat_z;
    target_pose.orientation.w = quat_w;
    
    waypoints.push_back(target_pose);

    moveit_msgs::msg::RobotTrajectory trajectory;
    double fraction = move_group_robot_->computeCartesianPath(
        waypoints, 0.01, 0.0, trajectory);

    if (fraction >= 0.9) {
      move_group_robot_->execute(trajectory);
    } else {
      RCLCPP_ERROR(LOGGER, "Cartesian Trajectory Failed!");
    }
  }

  void plan_trajectory_kinematics() {
    plan_success_robot_ = (move_group_robot_->plan(kinematics_trajectory_plan_) == moveit::core::MoveItErrorCode::SUCCESS);
  }

  void execute_trajectory_kinematics() {
    if (plan_success_robot_) move_group_robot_->execute(kinematics_trajectory_plan_);
  }

  void setup_joint_value_gripper(float angle) {
    joint_group_positions_gripper_[2] = angle;
    move_group_gripper_->setJointValueTarget(joint_group_positions_gripper_);
  }

  void plan_trajectory_gripper() {
    plan_success_gripper_ = (move_group_gripper_->plan(gripper_trajectory_plan_) == moveit::core::MoveItErrorCode::SUCCESS);
  }

  void execute_trajectory_gripper() {
    if (plan_success_gripper_) move_group_gripper_->execute(gripper_trajectory_plan_);
  }
}; 

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  std::shared_ptr<rclcpp::Node> base_node =
      std::make_shared<rclcpp::Node>("pick_and_place");
  PickAndPlace pick_and_place_node(base_node);
  pick_and_place_node.execute_trajectory_plan();
  rclcpp::shutdown();
  return 0;
}