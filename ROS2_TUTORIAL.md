# RGBTrack ROS2 Node — Launch & Visualization Tutorial

## Prerequisites

- ROS2 Jazzy installed at `/opt/ros/jazzy`
- Python dependencies installed (same as ZMQ mode)
- A valid `config.yaml` in the working directory

---

## 0. Python Version — Conda Environment Compatibility

ROS2's C extensions (`rclpy`, `_rclpy_pybind11`) are compiled against the **system Python**
(check which version: `ls /opt/ros/jazzy/lib/python*/`).
If you see `python3.10` there but your conda env uses Python 3.12, you will get:

```
ModuleNotFoundError: No module named 'rclpy._rclpy_pybind11'
The C extension '...cpython-310-x86_64-linux-gnu.so' isn't present on the system.
```

### Fix: run with the system Python that matches ROS2

```bash
# Deactivate conda so system python3.10 is used
conda deactivate

source /opt/ros/jazzy/setup.zsh
python3.10 main.py --backend ros2
```

### If your ML packages are only in conda

Install the inference dependencies for the system Python as well, **or** use a wrapper
that prepends the ROS2 packages to the system Python path used by the project:

```bash
conda deactivate
source /opt/ros/jazzy/setup.zsh

# Install project deps under system python3.10 (one-time)
python3.10 -m pip install --user -r requirements.txt

python3.10 main.py --backend ros2
```

> **Why not stay in conda?**
> ROS2 C extensions are ABI-linked to a specific CPython version. Conda's Python binary
> is a different build than the system one, so even if the version numbers match the
> `.so` files will fail to load. The cleanest solution is always to use the system
> Python that ROS2 was compiled against.

---

## 1. Source ROS2 Environment

Add this to your shell session (or `.zshrc` / `.bashrc`):

```bash
source /opt/ros/jazzy/setup.zsh   # zsh
# or
source /opt/ros/jazzy/setup.bash  # bash
```

---

## 2. Configure `config.yaml`

The `ros2` section controls the node name, namespace, and publish rate.
Add or edit this block in your `config.yaml`:

```yaml
ros2:
  node_name: rgbtrack          # ROS2 node name  → /rgbtrack
  namespace: ""                # Optional namespace prefix (e.g. "robot1" → /robot1/rgbtrack)
  publish_interval_ms: 0       # 0 = publish every new result; >0 = fixed rate (ms)
```

---

## 3. Launch the Node

```bash
# From the RGBTrack directory
# Deactivate conda first (see section 0), then:
conda deactivate
source /opt/ros/jazzy/setup.zsh
python3.10 main.py --backend ros2
```

With a custom config path:

```bash
python3.10 main.py --backend ros2 --config /path/to/my_config.yaml
```

You should see log output like:

```
INFO  Publisher service initialized (backend: ros2)
INFO  ROS2 node 'rgbtrack' started (namespace: '')
```

---

## 4. Verify the Node is Running

In a separate terminal (also sourced):

```bash
source /opt/ros/jazzy/setup.zsh

# List nodes
ros2 node list
# → /rgbtrack

# List topics
ros2 topic list
# → /rgbtrack/pose
# → /rgbtrack/twist
# → /rgbtrack/camera_info
# → /rgbtrack/status
# → /rgbtrack/preview_image

# List services
ros2 service list
# → /rgbtrack/start
# → /rgbtrack/pause
# → /rgbtrack/resume
# → /rgbtrack/reset
# → /rgbtrack/enable_frame_buffer
# → /rgbtrack/get_status
```

---

## 5. Control the Node

The node starts in **IDLE** state. Use services to drive the state machine:

```
IDLE  ──/start──▶  DETECTING  ──(auto)──▶  TRACKING
                                              │    ▲
                                         /pause  /resume
                                              ▼    │
                                            PAUSED
  (any state) ──/reset──▶ IDLE
```

### Start detection

```bash
ros2 service call /rgbtrack/start std_srvs/srv/Trigger
```

### Pause / Resume tracking

```bash
ros2 service call /rgbtrack/pause  std_srvs/srv/Trigger
ros2 service call /rgbtrack/resume std_srvs/srv/Trigger
```

### Reset to IDLE

```bash
ros2 service call /rgbtrack/reset std_srvs/srv/Trigger
```

### Query current status

```bash
ros2 service call /rgbtrack/get_status std_srvs/srv/Trigger
# Response: message field contains JSON:
# {"status": "TRACKING", "prompt": "red cup", "nms_threshold": 0.4, ...}
```

---

## 6. Set Parameters

### Change the CLIP detection prompt

```bash
ros2 param set /rgbtrack prompt "blue mug"
```

### Adjust NMS threshold

```bash
ros2 param set /rgbtrack nms_threshold 0.5
```

### View all current parameters

```bash
ros2 param list /rgbtrack
ros2 param get  /rgbtrack prompt
ros2 param get  /rgbtrack nms_threshold
```

---

## 7. Enable Camera Preview Stream

The `preview_image` topic is only published when the frame buffer is enabled:

```bash
ros2 service call /rgbtrack/enable_frame_buffer \
    std_srvs/srv/SetBool "{data: true}"
```

Disable it again to reduce bandwidth:

```bash
ros2 service call /rgbtrack/enable_frame_buffer \
    std_srvs/srv/SetBool "{data: false}"
```

---

## 8. Inspect Topic Data

```bash
# Stream pose (position + orientation quaternion)
ros2 topic echo /rgbtrack/pose

# Stream velocity
ros2 topic echo /rgbtrack/twist

# Stream status JSON
ros2 topic echo /rgbtrack/status

# Measure publish rate
ros2 topic hz /rgbtrack/pose
```

---

## 9. Visualize in RViz2

### Launch RViz2

```bash
rviz2
```

### Set the Fixed Frame

In the **Global Options** panel (top-left), set **Fixed Frame** to:

```
camera
```

This matches the `header.frame_id = "camera"` set by the node. All topics will then display correctly relative to the camera origin.

> **Note:** If you have a TF publisher providing a `camera` → `world` transform in your system,
> set **Fixed Frame** to your world frame instead (e.g. `world` or `base_link`).

### Add Displays

Click **Add** (bottom-left of RViz2) and add the following:

#### Object Pose (6D)

| Field | Value |
|---|---|
| Display Type | **Pose** |
| Topic | `/rgbtrack/pose` |
| Shape | **Axes** (recommended — shows X/Y/Z orientation clearly) |
| Shaft Length / Head Length | adjust to match object scale |

#### Object Velocity

| Field | Value |
|---|---|
| Display Type | **TwistStamped** (search for "Twist") |
| Topic | `/rgbtrack/twist` |
| Scale | 0.1–1.0 depending on expected velocity magnitude |

#### Camera Image (when frame buffer enabled)

| Field | Value |
|---|---|
| Display Type | **Image** |
| Topic | `/rgbtrack/preview_image` |

This opens a floating image panel showing the live camera feed.

#### Camera Info

| Field | Value |
|---|---|
| Display Type | **Camera** |
| Topic | `/rgbtrack/camera_info` |

This renders a camera frustum in the 3D view, helpful for understanding the sensor frame.

### Suggested RViz2 Layout

```
┌─────────────────────────────────────────────────────┐
│  3D View (Fixed Frame: camera)                      │
│    • Axes display at origin (camera frame)          │
│    • Pose display → /rgbtrack/pose (Axes shape)     │
│    • TwistStamped → /rgbtrack/twist                 │
│    • Camera frustum → /rgbtrack/camera_info         │
├─────────────────────────────────────────────────────┤
│  Image panel → /rgbtrack/preview_image              │
└─────────────────────────────────────────────────────┘
```

### Save RViz2 Config

Once set up, save your display config:
**File → Save Config As** → e.g. `rgbtrack_rviz.rviz`

Launch RViz2 with it next time:

```bash
rviz2 -d rgbtrack_rviz.rviz
```

---

## 10. Record Data with ros2 bag

Use `ros2 bag` to record all topics (replaces the ZMQ backend's built-in recording):

```bash
# Record all RGBTrack topics
ros2 bag record \
    /rgbtrack/pose \
    /rgbtrack/twist \
    /rgbtrack/camera_info \
    /rgbtrack/status \
    /rgbtrack/preview_image \
    -o rgbtrack_session

# Or record everything under the node
ros2 bag record /rgbtrack/ -o rgbtrack_session
```

Playback:

```bash
ros2 bag play rgbtrack_session
```

---

## 11. Quick-Start Checklist

```bash
# Terminal 1 — run the node (deactivate conda first)
conda deactivate
source /opt/ros/jazzy/setup.zsh
python3.10 main.py --backend ros2

# Terminal 2 — control
source /opt/ros/jazzy/setup.zsh
ros2 param set /rgbtrack prompt "red cup"
ros2 service call /rgbtrack/start std_srvs/srv/Trigger
ros2 service call /rgbtrack/enable_frame_buffer std_srvs/srv/SetBool "{data: true}"

# Terminal 3 — visualize
source /opt/ros/jazzy/setup.zsh
rviz2 -d rgbtrack_rviz.rviz
```
