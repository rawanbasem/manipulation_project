#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_msgs/msg/display_robot_state.hpp>
#include <moveit_msgs/msg/display_trajectory.hpp>
#include <chrono>
#include <cmath>
#include <memory>
#include <thread>
#include <vector>

static const rclcpp::Logger LOGGER = rclcpp::get_logger("move_group_node");
static const std::string PLANNING_GROUP_ROBOT = "ur_manipulator";
static const std::string PLANNING_GROUP_GRIPPER = "gripper";

class PickAndPlaceTrajectory {
public:
  PickAndPlaceTrajectory(rclcpp::Node::SharedPtr base_node_)
      : base_node_(base_node_) {
    RCLCPP_INFO(LOGGER, "Initializing Class: Pick And Place Trajectory...");

    // 1. Add the parameter-loaded base_node_
    executor_.add_node(base_node_);
    std::thread([this]() { this->executor_.spin(); }).detach();

    // 2. Initialize MoveGroupInterfaces
    move_group_robot_ = std::make_shared<MoveGroupInterface>(base_node_, PLANNING_GROUP_ROBOT);
    move_group_gripper_ = std::make_shared<MoveGroupInterface>(base_node_, PLANNING_GROUP_GRIPPER);

    // 3. Setup planning configurations
    joint_model_group_robot_ = move_group_robot_->getCurrentState()->getJointModelGroup(PLANNING_GROUP_ROBOT);
    joint_model_group_gripper_ = move_group_gripper_->getCurrentState()->getJointModelGroup(PLANNING_GROUP_GRIPPER);

    current_state_robot_ = move_group_robot_->getCurrentState(10);
    current_state_robot_->copyJointGroupPositions(joint_model_group_robot_, joint_group_positions_robot_);
    current_state_gripper_ = move_group_gripper_->getCurrentState(10);
    current_state_gripper_->copyJointGroupPositions(joint_model_group_gripper_, joint_group_positions_gripper_);

    move_group_robot_->setStartStateToCurrentState();
    move_group_gripper_->setStartStateToCurrentState();

    RCLCPP_INFO(LOGGER, "Class Initialized: Pick And Place Trajectory");
  }

  ~PickAndPlaceTrajectory() {
    RCLCPP_INFO(LOGGER, "Class Terminated: Pick And Place Trajectory");
  }

  void execute_trajectory_plan() {
    RCLCPP_INFO(LOGGER, "Executing Hover and Gripper Open Sequence...");

    // STEP 1: (Pregrasp) position
    RCLCPP_INFO(LOGGER, "Moving to Hover Position...");
    setup_goal_pose_target(5.280, -3.840, 1.042, -1.0, 0.0, 0.0, 0.0);
    plan_trajectory_kinematics();
    execute_trajectory_kinematics();

    // STEP 2: Open the gripper
    RCLCPP_INFO(LOGGER, "Opening Gripper...");
    setup_named_pose_gripper("open");
    plan_trajectory_gripper();
    execute_trajectory_gripper();

    RCLCPP_INFO(LOGGER, "Hover & Open Sequence Complete!");
  }

private:
  using MoveGroupInterface = moveit::planning_interface::MoveGroupInterface;
  using JointModelGroup = moveit::core::JointModelGroup;
  using RobotStatePtr = moveit::core::RobotStatePtr;
  using Plan = MoveGroupInterface::Plan;
  using Pose = geometry_msgs::msg::Pose;

  rclcpp::Node::SharedPtr base_node_;
  rclcpp::executors::SingleThreadedExecutor executor_;

  std::shared_ptr<MoveGroupInterface> move_group_robot_;
  std::shared_ptr<MoveGroupInterface> move_group_gripper_;

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

  void setup_goal_pose_target(float pos_x, float pos_y, float pos_z,
                              float quat_x, float quat_y, float quat_z, float quat_w) {
    target_pose_robot_.position.x = pos_x;
    target_pose_robot_.position.y = pos_y;
    target_pose_robot_.position.z = pos_z;
    target_pose_robot_.orientation.x = quat_x;
    target_pose_robot_.orientation.y = quat_y;
    target_pose_robot_.orientation.z = quat_z;
    target_pose_robot_.orientation.w = quat_w;
    move_group_robot_->setPoseTarget(target_pose_robot_);
  }

  void plan_trajectory_kinematics() {
    plan_success_robot_ = (move_group_robot_->plan(kinematics_trajectory_plan_) == moveit::core::MoveItErrorCode::SUCCESS);
  }

  void execute_trajectory_kinematics() {
    if (plan_success_robot_) {
      move_group_robot_->execute(kinematics_trajectory_plan_);
      RCLCPP_INFO(LOGGER, "Robot Hover Trajectory Success !");
    } else {
      RCLCPP_ERROR(LOGGER, "Robot Hover Trajectory Failed !");
    }
  }

  void setup_named_pose_gripper(std::string_view pose_name) {
    move_group_gripper_->setNamedTarget(std::string(pose_name));
  }

  void plan_trajectory_gripper() {
    plan_success_gripper_ = (move_group_gripper_->plan(gripper_trajectory_plan_) == moveit::core::MoveItErrorCode::SUCCESS);
  }

  void execute_trajectory_gripper() {
    if (plan_success_gripper_) {
      move_group_gripper_->execute(gripper_trajectory_plan_);
      RCLCPP_INFO(LOGGER, "Gripper Open Command Success !");
    } else {
      RCLCPP_ERROR(LOGGER, "Gripper Open Command Failed !");
    }
  }
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  std::shared_ptr<rclcpp::Node> base_node = std::make_shared<rclcpp::Node>("pick_and_place_trajectory");
  PickAndPlaceTrajectory pick_and_place_trajectory_node(base_node);
  pick_and_place_trajectory_node.execute_trajectory_plan();
  rclcpp::shutdown();
  return 0;
}