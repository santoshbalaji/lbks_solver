"""ROS bridge for the RViz panel, target gizmo, and background IK worker."""

from concurrent.futures import ThreadPoolExecutor
import json
import time

from geometry_msgs.msg import Point, PointStamped
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import JointState
from std_msgs.msg import String
import torch
from visualization_msgs.msg import (
    InteractiveMarker, InteractiveMarkerControl, InteractiveMarkerFeedback,
    Marker, MarkerArray,
)

from lbks_gui.engine import PreviewEngine
from lbks_gui.robot import load_robot, load_variants


class PreviewNode(Node):
    def __init__(self):
        super().__init__('lbks_ik_preview')
        self.declare_parameter('urdf_path', '')
        self.declare_parameter('models_config', '')
        self.declare_parameter('tip_link', 'tool0')
        urdf, self.frame = load_robot(self.get_parameter('urdf_path').value)
        self.tip = self.get_parameter('tip_link').value
        self.engine = PreviewEngine(urdf, self.tip)
        self.variants = load_variants(self.get_parameter('models_config').value)
        self.joints = [0.0] * len(self.engine.names)
        self.target = [0.4, 0.0, 0.4]
        self.actual = self.engine.fk(self.joints).tolist()
        self.model_name, self.custom_path = 'numerical', ''
        self.status = 'Ready — choose a target or drag the XYZ gizmo.'
        self.obstacle = False
        self.elapsed_ms = 0.0
        self.generation = 0
        self.pending = None
        self.future = None
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.joint_pub = self.create_publisher(JointState, '/lbks_gui/joint_states', 10)
        self.state_pub = self.create_publisher(String, '/lbks_gui/state', 10)
        self.trace_pub = self.create_publisher(
            String, '/lbks_gui/network_trace',
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.publish_trace('Load a trained checkpoint and solve a target to inspect its network.')
        self.marker_pub = self.create_publisher(MarkerArray, '/lbks_gui/markers', 10)
        self.create_subscription(String, '/lbks_gui/request', self.request, 10)
        self.create_subscription(PointStamped, '/clicked_point', self.clicked, 10)
        self.server = InteractiveMarkerServer(self, 'lbks_gui/target')
        self.make_target()
        self.create_timer(0.05, self.tick)

    def make_target(self):
        marker = InteractiveMarker()
        marker.header.frame_id = self.frame
        marker.name = 'target'
        marker.description = 'Cartesian target (m)'
        marker.scale = 0.22
        marker.pose.position = Point(x=self.target[0], y=self.target[1], z=self.target[2])
        marker.pose.orientation.w = 1.0
        for name, axis in [('X', (1., 0., 0.)), ('Y', (0., 0., 1.)), ('Z', (0., 1., 0.))]:
            control = InteractiveMarkerControl()
            control.name = name
            control.orientation.w = 0.70710678
            control.orientation.x, control.orientation.y, control.orientation.z = (
                value * 0.70710678 for value in axis)
            control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
            marker.controls.append(control)
        self.server.insert(marker, feedback_callback=self.feedback)
        self.server.applyChanges()

    def feedback(self, feedback):
        if feedback.event_type == InteractiveMarkerFeedback.MOUSE_UP:
            p = feedback.pose.position
            self.enqueue([p.x, p.y, p.z])

    def clicked(self, point):
        if point.header.frame_id == self.frame:
            self.enqueue([point.point.x, point.point.y, point.point.z])
        else:
            self.status = f'Pick a point in the {self.frame} fixed frame.'

    def enqueue(self, target):
        values = torch.as_tensor(target, dtype=torch.float32)
        if values.shape != (3,) or not torch.isfinite(values).all():
            raise ValueError('Expected three finite target coordinates')
        self.target = values.tolist()
        self.generation += 1
        self.pending = (self.generation, list(self.target), list(self.joints),
                        self.model_name, self.custom_path)
        self.status = 'Solving…'
        self.publish_trace('Prediction in progress…')
        self.make_target()

    def request(self, message):
        try:
            data = json.loads(message.data)
            operation = data.get('op', 'solve')
            if operation == 'solve':
                name = data.get('model', self.model_name)
                if name not in ('numerical', '__custom__') and name not in self.variants:
                    raise ValueError(f'Unknown model: {name}')
                self.model_name = name
                self.custom_path = data.get('path', self.custom_path)
                self.enqueue(data['target'])
            elif operation in ('joints', 'home'):
                q = torch.as_tensor(data['joints'] if operation == 'joints' else
                                    [0.] * len(self.joints), dtype=torch.float32)
                if q.shape != (len(self.joints),) or not torch.isfinite(q).all():
                    raise ValueError('Invalid joint vector')
                self.generation += 1
                self.pending = None
                self.joints = q.clamp(self.engine.lower, self.engine.upper).tolist()
                self.actual = self.engine.fk(self.joints).tolist()
                self.status = 'Manual joint preview'
                self.publish_trace('Manual joint preview: no network prediction.')
            elif operation == 'obstacle':
                self.obstacle = bool(data['enabled'])
            else:
                raise ValueError(f'Unknown operation: {operation}')
        except Exception as exc:
            self.status = f'Error: {exc}'
            self.publish_trace(self.status)

    def publish_trace(self, note, trace=None):
        payload = dict(trace) if trace is not None else {'mode': 'idle', 'stages': [], 'note': note}
        payload.update(prediction_id=self.generation, model=self.model_name,
                       checkpoint=self.custom_path if self.model_name == '__custom__' else '',
                       target=self.target, elapsed_ms=self.elapsed_ms)
        self.trace_pub.publish(String(data=json.dumps(payload, allow_nan=False)))

    def solve(self, target, seed, name, path):
        start = time.perf_counter()
        spec = None if name == 'numerical' else (
            {'path': path} if name == '__custom__' else self.variants[name])
        result = self.engine.solve(target, seed, spec)
        result['elapsed_ms'] = (time.perf_counter() - start) * 1000
        return result

    def tick(self):
        if self.future is not None and self.future.done():
            try:
                result = self.future.result()
                if self.running_generation == self.generation:
                    self.joints, self.actual = result['joints'], result['actual']
                    self.elapsed_ms = result['elapsed_ms']
                    self.status = ('Target reached' if result['error'] < 0.01 else
                                   'Approximate pose — target not reached')
                    if result['clipped']:
                        self.status += ' (joint limits applied)'
                    self.publish_trace('', result['network_trace'])
            except Exception as exc:
                if self.running_generation == self.generation:
                    self.status = f'Error: {exc}'
                    self.publish_trace(self.status)
            self.future = None
        if self.future is None and self.pending:
            self.running_generation, *args = self.pending
            self.pending = None
            self.future = self.pool.submit(self.solve, *args)
        now = self.get_clock().now().to_msg()
        joints = JointState()
        joints.header.stamp = now
        joints.name, joints.position = self.engine.names, self.joints
        self.joint_pub.publish(joints)
        error = sum((a - b) ** 2 for a, b in zip(self.target, self.actual)) ** 0.5
        state = dict(names=self.engine.names, lower=self.engine.lower.tolist(),
                     upper=self.engine.upper.tolist(), joints=self.joints,
                     target=self.target, actual=self.actual, error=error,
                     status=self.status, busy=self.future is not None,
                     models=list(self.variants), frame=self.frame, tip=self.tip,
                     elapsed_ms=self.elapsed_ms)
        self.state_pub.publish(String(data=json.dumps(state)))
        markers = []
        for ident, position, color in [(0, self.target, (0.2, 0.7, 1.)),
                                        (1, self.actual, (0.2, 1., 0.6))]:
            marker = self.marker(ident, Marker.SPHERE, color)
            marker.pose.position = Point(x=position[0], y=position[1], z=position[2])
            marker.scale.x = marker.scale.y = marker.scale.z = 0.035
            markers.append(marker)
        line = self.marker(2, Marker.LINE_LIST, (1., 0.65, 0.2))
        line.scale.x = 0.006
        line.points = [Point(x=p[0], y=p[1], z=p[2]) for p in (self.target, self.actual)]
        markers.append(line)
        obstacle = self.marker(3, Marker.CUBE, (1., 0.4, 0.25))
        obstacle.action = Marker.ADD if self.obstacle else Marker.DELETE
        obstacle.pose.position = Point(x=0.4, y=0.2, z=0.25)
        obstacle.scale.x, obstacle.scale.y, obstacle.scale.z = 0.2, 0.2, 0.5
        obstacle.color.a = 0.35
        markers.append(obstacle)
        self.marker_pub.publish(MarkerArray(markers=markers))

    def marker(self, ident, kind, color):
        marker = Marker()
        marker.header.frame_id = self.frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns, marker.id, marker.type = 'preview', ident, kind
        marker.pose.orientation.w = 1.0
        marker.color.r, marker.color.g, marker.color.b = color
        marker.color.a = 1.0
        return marker

    def destroy_node(self):
        self.pool.shutdown(wait=True, cancel_futures=True)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        torch.set_num_threads(1)
        node = PreviewNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()
