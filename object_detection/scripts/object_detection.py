#!/usr/bin/env python3
import sys
from pathlib import Path

# Remove script directory from sys.path to prevent module name collision
script_dir = str(Path(__file__).resolve().parent)
if script_dir in sys.path:
    sys.path.remove(script_dir)

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from object_detection.msg import DetectedSurfaces, DetectedObjects

import pcl
import numpy as np
import tf2_ros
from tf2_ros import TransformException, ConnectivityException
from typing import List, Tuple, Union


class ObjectDetection(Node):
    def __init__(self) -> None:
        super().__init__('object_detection_node')

        self.pc_sub = self.create_subscription(
            PointCloud2,
            '/wrist_rgbd_depth_sensor/points',
            self.callback,
            10
        )
        self.surface_pub = self.create_publisher(MarkerArray, 'table_markers', 10)
        self.surface_detected_pub = self.create_publisher(DetectedSurfaces, 'surface_detected', 10)
        self.objects_pub = self.create_publisher(MarkerArray, 'object_markers', 10)
        self.object_detected_pub = self.create_publisher(DetectedObjects, 'object_detected', 10)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.target_frame = 'base_link'

    def callback(self, msg: PointCloud2) -> None:
        try:
            cloud = self.from_ros_msg(msg)
            if cloud is None or cloud.size == 0:
                return

            # Surface Filter: broad workspace filter around table height 
            filtered_cloud_plane = self.filter_cloud(
                cloud,
                min_x=0.10,
                max_x=0.70,
                min_y=-0.50,
                max_y=0.50,
                min_z=-0.05,
                max_z=0.01
            )

            # Object Filter: above table height where the block sits 
            filtered_cloud_objects = self.filter_cloud(
                cloud,
                min_x=0.10,
                max_x=0.70,
                min_y=-0.50,
                max_y=0.50,
                min_z=0.012,
                max_z=0.08
            )

            # 1. Process table surface
            if filtered_cloud_plane is not None and filtered_cloud_plane.size > 50:
                plane_indices, plane_coefficients, plane_cloud = self.extract_plane(filtered_cloud_plane)
                if plane_cloud is not None and plane_cloud.size > 50:
                    _, surface_centroids, surface_dimensions = self.extract_clusters(
                        plane_cloud, "Table Surface", min_size=50
                    )
                    self.pub_surface_marker(surface_centroids, surface_dimensions)
                    self.pub_surface_detected(surface_centroids, surface_dimensions)

            # 2. Process object (cube)
            if filtered_cloud_objects is not None and filtered_cloud_objects.size > 15:
                _, object_centroids, object_dimensions = self.extract_clusters(
                    filtered_cloud_objects, "Object", min_size=15
                )
                self.pub_object_marker(object_centroids, object_dimensions)
                self.pub_object_detected(object_centroids, object_dimensions)

        except (TransformException, ConnectivityException) as e:
            self.get_logger().error(f"Transform lookup failed: {e}")
        except Exception as e:
            self.get_logger().error(f"Error in callback: {e}")

    def from_ros_msg(self, msg: PointCloud2) -> Union[pcl.PointCloud, None]:
        """Vectorized conversion & base_link TF transformation."""
        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                msg.header.frame_id,
                rclpy.time.Time(),
                timeout=rclpy.time.Duration(seconds=0.5)
            )
            trans = np.array([
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z
            ], dtype=np.float32)

            quat = np.array([
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w
            ], dtype=np.float32)

            rot = self.quaternion_to_rotation_matrix(quat)

            # Unpack XYZ data using numpy
            raw_data = np.frombuffer(msg.data, dtype=np.uint8)
            num_points = len(raw_data) // msg.point_step
            reshaped = raw_data[:num_points * msg.point_step].reshape((num_points, msg.point_step))
            xyz_points = np.frombuffer(np.ascontiguousarray(reshaped[:, :12]), dtype=np.float32).reshape(-1, 3)

            # Filter non-finite points (NaN, Inf)
            valid_mask = np.isfinite(xyz_points).all(axis=1)
            valid_points = xyz_points[valid_mask]

            if valid_points.shape[0] == 0:
                return None

            # Matrix multiplication to rotate and shift to base_link: P_base = P_cam * R^T + T
            transformed_points = np.dot(valid_points, rot.T) + trans

            cloud = pcl.PointCloud()
            cloud.from_array(transformed_points.astype(np.float32))
            return cloud

        except Exception:
            return None

    def quaternion_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        x, y, z, w = q
        return np.array([
            [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
            [2*x*y + 2*z*w,     1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x**2 - 2*y**2]
        ], dtype=np.float32)

    def filter_cloud(self, cloud: pcl.PointCloud,
                     min_x: float, max_x: float,
                     min_y: float, max_y: float,
                     min_z: float, max_z: float) -> Union[pcl.PointCloud, None]:
        """NumPy range slicing on X, Y, and Z axes."""
        try:
            pts = cloud.to_array()
            mask = (
                (pts[:, 0] >= min_x) & (pts[:, 0] <= max_x) &
                (pts[:, 1] >= min_y) & (pts[:, 1] <= max_y) &
                (pts[:, 2] >= min_z) & (pts[:, 2] <= max_z)
            )
            filtered_pts = pts[mask]

            if filtered_pts.shape[0] == 0:
                return None

            filtered_cloud = pcl.PointCloud()
            filtered_cloud.from_array(filtered_pts.astype(np.float32))
            return filtered_cloud
        except Exception as e:
            self.get_logger().error(f"Error in filter_cloud: {e}")
            return None

    def extract_plane(self, cloud: pcl.PointCloud) -> Tuple[np.ndarray, np.ndarray, pcl.PointCloud]:
        if cloud.size < 10:
            return np.array([]), np.array([]), None

        seg = cloud.make_segmenter()
        seg.set_model_type(pcl.SACMODEL_PLANE)
        seg.set_method_type(pcl.SAC_RANSAC)
        seg.set_distance_threshold(0.01)
        indices, coefficients = seg.segment()

        if len(indices) == 0:
            return np.array([]), np.array([]), None

        plane_cloud = cloud.extract(indices)
        return indices, coefficients, plane_cloud

    def extract_clusters(self, cloud: pcl.PointCloud, cluster_type: str, min_size: int = 20) -> Tuple[List[pcl.PointCloud], List[List[float]], List[List[float]]]:
        if cloud is None or cloud.size < min_size:
            return [], [], []

        tree = cloud.make_kdtree()
        ec = cloud.make_EuclideanClusterExtraction()
        ec.set_ClusterTolerance(0.02)
        ec.set_MinClusterSize(min_size)
        ec.set_MaxClusterSize(80000)
        ec.set_SearchMethod(tree)
        cluster_indices = ec.Extract()

        clusters, centroids, dimensions = [], [], []

        for indices in cluster_indices:
            cluster = cloud.extract(indices)
            cluster_np = cluster.to_array()

            centroid = np.mean(cluster_np, axis=0).tolist()
            min_coords = np.min(cluster_np, axis=0)
            max_coords = np.max(cluster_np, axis=0)
            dim = (max_coords - min_coords).tolist()

            clusters.append(cluster)
            centroids.append(centroid)
            dimensions.append(dim)

        return clusters, centroids, dimensions

    def pub_surface_marker(self, surface_centroids: List[List[float]], surface_dimensions: List[List[float]]) -> None:
        marker_array = MarkerArray()
        surface_thickness = 0.05

        for idx, (centroid, dimensions) in enumerate(zip(surface_centroids, surface_dimensions)):
            cube_marker = Marker()
            cube_marker.header.frame_id = self.target_frame
            cube_marker.header.stamp = self.get_clock().now().to_msg()
            cube_marker.id = idx
            cube_marker.type = Marker.CUBE
            cube_marker.action = Marker.ADD
            cube_marker.pose.position.x = float(centroid[0])
            cube_marker.pose.position.y = float(centroid[1])
            cube_marker.pose.position.z = float(centroid[2] - surface_thickness / 2.0)
            cube_marker.pose.orientation.w = 1.0

            cube_marker.scale.x = float(dimensions[0])
            cube_marker.scale.y = float(dimensions[1])
            cube_marker.scale.z = float(surface_thickness)

            cube_marker.color.r = 0.0
            cube_marker.color.g = 1.0
            cube_marker.color.b = 0.0
            cube_marker.color.a = 0.5
            marker_array.markers.append(cube_marker)

        if marker_array.markers:
            self.surface_pub.publish(marker_array)

    def pub_object_marker(self, object_centroids: List[List[float]], object_dimensions: List[List[float]]) -> None:
        marker_array = MarkerArray()

        for idx, (centroid, dimensions) in enumerate(zip(object_centroids, object_dimensions)):
            marker = Marker()
            marker.header.frame_id = self.target_frame
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.id = idx
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = float(centroid[0])
            marker.pose.position.y = float(centroid[1])
            marker.pose.position.z = float(centroid[2])
            marker.pose.orientation.w = 1.0

            marker.scale.x = float(dimensions[0])
            marker.scale.y = float(dimensions[1])
            marker.scale.z = float(dimensions[2])

            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.7
            marker_array.markers.append(marker)

        if marker_array.markers:
            self.objects_pub.publish(marker_array)

    def pub_surface_detected(self, centroids: List[List[float]], dimensions: List[List[float]]) -> None:
        for idx, (centroid, dimension) in enumerate(zip(centroids, dimensions)):
            surface_msg = DetectedSurfaces()
            surface_msg.surface_id = int(idx)
            surface_msg.position.x = float(centroid[0])
            surface_msg.position.y = float(centroid[1])
            surface_msg.position.z = float(centroid[2])
            surface_msg.height = float(dimension[0])
            surface_msg.width = float(dimension[1])
            self.surface_detected_pub.publish(surface_msg)

    def pub_object_detected(self, centroids: List[List[float]], dimensions: List[List[float]]) -> None:
        for idx, (centroid, dimension) in enumerate(zip(centroids, dimensions)):
            object_msg = DetectedObjects()
            object_msg.object_id = int(idx)
            object_msg.position.x = float(centroid[0])
            object_msg.position.y = float(centroid[1])
            object_msg.position.z = float(centroid[2])
            object_msg.height = float(dimension[0])
            object_msg.width = float(dimension[1])
            object_msg.thickness = float(dimension[2])
            self.object_detected_pub.publish(object_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ObjectDetection()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()