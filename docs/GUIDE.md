# RGBTrack Complete Guide

RGBTrack is a multi-threaded Python framework for camera-based detection with real-time configuration and ZeroMQ publishing.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture Overview](#architecture-overview)
3. [Configuration](#configuration)
4. [ZeroMQ Communication](#zeromq-communication)
5. [API Reference](#api-reference)
6. [Development](#development)

---

## Quick Start

### Installation

```bash
# Clone repository
cd /path/to/RGBTrack

# Create UV environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Running the Application

```bash
# Start application (creates config.yaml if missing)
python main.py
```

The application will:
- Load configuration from `config.yaml` (or create default)
- Start camera interface
- Launch Web UI at http://localhost:7860
- Start detection loop
- Begin ZMQ publisher on Unix socket `/tmp/rgbtrack.sock`

### Subscribing to Results

```bash
# Standard subscriber
python -m src.zmq_subscriber

# Enhanced subscriber with statistics
python -m src.zmq_subscriber --enhanced

# TCP subscriber (if configured)
python -m src.zmq_subscriber --transport tcp --host localhost --port 5555
```

---

## Architecture Overview

### Multi-threaded Design

```
┌──────────────────────────────────────────────────────────────┐
│                     RGBTrackApplication                       │
│                   (Main Thread Coordinator)                   │
└──────────────────────────────────────────────────────────────┘
                             │
           ┌─────────────────┼─────────────────┐
           ▼                 ▼                 ▼
    ┌────────────┐   ┌────────────┐   ┌────────────┐
    │ Thread 1:  │   │ Thread 2:  │   │ Thread 3:  │
    │  Web UI    │   │ Detection  │   │ ZMQ Publish│
    │  (Gradio)  │   │   Loop     │   │            │
    └────────────┘   └────────────┘   └────────────┘
          │                 │                 │
          │                 ▼                 │
          │          ┌────────────┐           │
          │          │  Camera    │           │
          └─────────▶│  Interface │◀──────────┘
                     └────────────┘
                           │
                           ▼
                    ┌────────────┐
                    │ Detection  │
                    │ Algorithm  │
                    └────────────┘
```

### Thread Responsibilities

**Thread 1: Web UI (Gradio)**
- Serves configuration interface at http://localhost:7860
- Provides camera parameter controls (exposure, gain, white balance)
- Live camera preview
- Automatic configuration persistence

**Thread 2: Detection Loop**
- Continuous frame acquisition from camera
- Runs detection algorithm on each frame
- Maintains frame buffer and FPS statistics
- Pushes results to ZMQ publisher

**Thread 3: ZMQ Publisher**
- Listens for detection results
- Publishes results via Unix socket (IPC) or TCP
- Supports multiple subscribers
- Handles subscriber disconnection gracefully

### Component Interaction

```
┌──────────┐     Config Update      ┌──────────┐
│  WebUI   │───────────────────────▶│  Camera  │
└──────────┘                         └──────────┘
                                          │
                                    Frame │
                                          ▼
                                    ┌──────────┐
                      Results       │Detection │
┌──────────┐◀─────────────────────│ Algorithm│
│   ZMQ    │                        └──────────┘
│Publisher │
└────┬─────┘
     │ Publish
     │
     ▼
┌──────────────────────────────────┐
│  Subscribers (Multiple Allowed)  │
│  • Standard Subscriber            │
│  • Enhanced Subscriber (Stats)    │
│  • Custom Applications            │
└──────────────────────────────────┘
```

---

## Configuration

RGBTrack uses **Pydantic** for type-safe configuration and **YAML** for persistent storage.

### Configuration Structure

```yaml
# config.yaml
camera:
  device_id: 0
  exposure_time_ms: 30        # 1-100 ms
  gain: 1.0                    # 0.0-10.0
  red_balance: 1.0             # 0.5-2.0
  green_balance: 1.0           # 0.5-2.0
  blue_balance: 1.0            # 0.5-2.0
  mode: normal                 # normal | high_speed
  width: 1280
  height: 720

detection:
  confidence_threshold: 0.5
  nms_threshold: 0.4
  model_path: models/detector.pt

zmq:
  transport: ipc               # ipc | tcp
  socket_path: /tmp/rgbtrack.sock
  host: localhost              # Used for TCP
  port: 5555                   # Used for TCP

ui_host: 0.0.0.0
ui_port: 7860
max_fps: 30
frame_buffer_size: 10
```

### Pydantic Models

All configuration uses Pydantic BaseModel for validation:

```python
from src.config import SystemConfig

# Load from YAML (creates default if missing)
config = SystemConfig.from_yaml("config.yaml")

# Modify configuration
config.camera.exposure_time_ms = 50
config.zmq.transport = "tcp"

# Save to YAML
config.to_yaml("config.yaml")
```

### Configuration Hierarchy

```
SystemConfig (Root)
├── CameraConfig
│   ├── device_id: int
│   ├── exposure_time_ms: int
│   ├── gain: float
│   ├── red_balance: float
│   ├── green_balance: float
│   ├── blue_balance: float
│   ├── mode: Literal["normal", "high_speed"]
│   ├── width: int
│   └── height: int
├── DetectionConfig
│   ├── confidence_threshold: float
│   ├── nms_threshold: float
│   └── model_path: str
├── ZMQConfig
│   ├── transport: Literal["tcp", "ipc"]
│   ├── host: str
│   ├── port: int
│   ├── socket_path: str
│   └── address: str (property)
├── ui_host: str
├── ui_port: int
├── max_fps: int
└── frame_buffer_size: int
```

### Automatic Persistence

```
Start App → Load config.yaml (or create default)
    │
    ▼
Adjust settings in Web UI
    │
    ▼
Click "Apply" button
    │
    ▼
Changes applied to camera + Auto-save to config.yaml ✅
    │
    ▼
Restart App → Settings restored from config.yaml
```

### Using Configuration in Code

```python
from src.config import SystemConfig, CameraConfig, ZMQConfig

# Load existing configuration
config = SystemConfig.from_yaml("config.yaml")

# Access nested values
print(config.camera.exposure_time_ms)  # → 30
print(config.zmq.address)              # → "ipc:///tmp/rgbtrack.sock"

# Create custom configuration
custom_config = SystemConfig(
    camera=CameraConfig(
        device_id=1,
        exposure_time_ms=60,
        mode="high_speed"
    ),
    zmq=ZMQConfig(
        transport="tcp",
        host="0.0.0.0",
        port=6666
    )
)

# Save to file
custom_config.to_yaml("custom_config.yaml")
```

### Type Validation

Pydantic automatically validates types:

```python
config = SystemConfig()

# Valid
config.camera.exposure_time_ms = 50         # ✅ int
config.camera.gain = 2.5                     # ✅ float
config.camera.mode = "high_speed"            # ✅ Literal

# Invalid (raises ValidationError)
config.camera.exposure_time_ms = "invalid"  # ❌ TypeError
config.camera.mode = "turbo"                # ❌ Not in Literal
```

---

## ZeroMQ Communication

### Transport Options

| Transport | Use Case | Performance | Configuration |
|-----------|----------|-------------|---------------|
| **IPC** (Unix socket) | Same machine | ⚡ **BEST** - Direct IPC | `transport: ipc` |
| **TCP** | Network/Different machines | 🌐 Standard - Network overhead | `transport: tcp` |

### IPC Configuration (Recommended for Local)

```yaml
zmq:
  transport: ipc
  socket_path: /tmp/rgbtrack.sock
```

**Benefits:**
- Lower latency (~1.5-2x faster than TCP)
- Higher throughput
- Lower CPU usage
- No network configuration needed

### TCP Configuration (For Network)

```yaml
zmq:
  transport: tcp
  host: 0.0.0.0      # Listen on all interfaces
  port: 5555
```

**Use cases:**
- Subscribers on different machines
- Distributed systems
- Remote monitoring

### Publisher (Automatic)

The application automatically starts a ZMQ publisher based on configuration:

```python
from src.zmq_publisher import ZMQPublisher
from src.config import ZMQConfig

# Created automatically in RGBTrackApplication
config = ZMQConfig(transport="ipc", socket_path="/tmp/rgbtrack.sock")
publisher = ZMQPublisher(config)
publisher.start()

# Publish detection results
result = DetectionResult(
    timestamp=time.time(),
    frame_id=123,
    detections=[...],
    fps=30.0
)
publisher.publish(result)
```

### Subscriber Usage

#### Command-Line Subscriber

```bash
# Standard subscriber (default IPC)
python -m src.zmq_subscriber

# Enhanced subscriber with statistics
python -m src.zmq_subscriber --enhanced

# TCP subscriber
python -m src.zmq_subscriber --transport tcp --host localhost --port 5555

# Custom socket path
python -m src.zmq_subscriber --transport ipc --socket-path /tmp/custom.sock

# With timeout
python -m src.zmq_subscriber --timeout 5000
```

#### Programmatic Subscriber

```python
from src.zmq_subscriber import ZMQSubscriber, EnhancedSubscriber
from src.config import ZMQConfig

# Standard subscriber
config = ZMQConfig(transport="ipc", socket_path="/tmp/rgbtrack.sock")
subscriber = ZMQSubscriber(config)
subscriber.start()

# Receive results
while True:
    result = subscriber.receive()
    if result:
        print(f"Frame {result.frame_id}: {len(result.detections)} detections")

# Enhanced subscriber with statistics
subscriber = EnhancedSubscriber(config)
subscriber.start()
```

#### Custom Subscriber

```python
import zmq
import pickle

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("ipc:///tmp/rgbtrack.sock")
socket.setsockopt(zmq.SUBSCRIBE, b"")

while True:
    message = socket.recv()
    result = pickle.loads(message)
    
    # Process result
    print(f"Timestamp: {result.timestamp}")
    print(f"Frame ID: {result.frame_id}")
    print(f"Detections: {result.detections}")
    print(f"FPS: {result.fps}")
```

### Message Format

Messages are serialized using pickle:

```python
@dataclass
class DetectionResult:
    timestamp: float              # Unix timestamp
    frame_id: int                 # Sequential frame number
    detections: List[Detection]   # List of detected objects
    fps: float                    # Current processing FPS
```

### Multiple Subscribers

ZeroMQ PUB-SUB pattern supports multiple subscribers:

```bash
# Terminal 1: Publisher (automatic in main.py)
python main.py

# Terminal 2: Standard subscriber
python -m src.zmq_subscriber

# Terminal 3: Enhanced subscriber
python -m src.zmq_subscriber --enhanced

# Terminal 4: Custom application
python my_custom_subscriber.py
```

All subscribers receive the same messages simultaneously.

### Performance Comparison

#### IPC (Unix Socket)
```
Latency: ~50-100 μs
Throughput: ~5-10 million msg/s
CPU: Minimal overhead
```

#### TCP (localhost)
```
Latency: ~100-200 μs
Throughput: ~2-5 million msg/s
CPU: TCP/IP stack overhead
```

**Recommendation**: Use IPC for local subscribers, TCP for remote subscribers.

---

## API Reference

### RGBTrackApplication

Main application class that orchestrates all components.

```python
from src.app import RGBTrackApplication
from pathlib import Path

# Create application with default config.yaml
app = RGBTrackApplication()

# Or specify custom config
app = RGBTrackApplication(config_file=Path("custom_config.yaml"))

# Start all threads
app.start()

# Graceful shutdown
app.stop()
```

**Methods:**
- `start()` - Start all threads (UI, detection, ZMQ)
- `stop()` - Stop all threads gracefully
- `get_current_frame()` - Get latest camera frame
- `update_camera_config(config: CameraConfig)` - Update camera settings

### Camera Interface

```python
from src.camera import create_camera, CameraBase

# Create camera (auto-selects implementation)
camera = create_camera(device_id=0)

# Camera operations
camera.open()
frame = camera.read()
camera.apply_config(config)
camera.close()
```

**Implementations:**
- `DummyCamera` - Testing/development
- `MindVisionCamera` - Real MindVision SDK camera (when available)

### Detection Algorithm

```python
from src.detection import DetectionAlgorithm, DetectionResult

algorithm = DetectionAlgorithm(config.detection)
result = algorithm.detect(frame)

# Result contains:
# - timestamp: float
# - frame_id: int
# - detections: List[Detection]
# - fps: float
```

### ZMQ Publisher

```python
from src.zmq_publisher import ZMQPublisher

publisher = ZMQPublisher(config.zmq)
publisher.start()
publisher.publish(result)
publisher.stop()
```

### ZMQ Subscriber

```python
from src.zmq_subscriber import ZMQSubscriber, EnhancedSubscriber

# Standard
subscriber = ZMQSubscriber(config.zmq, timeout_ms=1000)
subscriber.start()
result = subscriber.receive()
subscriber.stop()

# Enhanced (with statistics)
subscriber = EnhancedSubscriber(config.zmq)
subscriber.start()
```

### Configuration

```python
from src.config import SystemConfig, CameraConfig, DetectionConfig, ZMQConfig

# Load/Save
config = SystemConfig.from_yaml("config.yaml")
config.to_yaml("config.yaml")

# Create programmatically
config = SystemConfig(
    camera=CameraConfig(device_id=0, exposure_time_ms=50),
    detection=DetectionConfig(confidence_threshold=0.7),
    zmq=ZMQConfig(transport="ipc", socket_path="/tmp/rgb.sock")
)

# Access properties
address = config.zmq.address  # Auto-computed based on transport
```

---

## Development

### Project Structure

```
RGBTrack/
├── src/
│   ├── __init__.py
│   ├── app.py              # Main application
│   ├── camera.py           # Camera interface
│   ├── config.py           # Pydantic configuration
│   ├── detection.py        # Detection algorithm
│   ├── webui.py            # Gradio Web UI
│   ├── zmq_publisher.py    # ZMQ publisher
│   └── zmq_subscriber.py   # ZMQ subscribers
├── docs/
│   └── GUIDE.md            # This file
├── config.yaml             # Runtime configuration (auto-generated)
├── config.example.yaml     # Example configuration
├── main.py                 # Entry point
└── requirements.txt        # Dependencies
```

### Dependencies

```txt
gradio>=4.0.0        # Web UI framework
pyzmq>=25.0.0        # ZeroMQ Python bindings
numpy>=1.24.0        # Numerical operations
opencv-python>=4.8.0 # Image processing
Pillow>=10.0.0       # Image handling
pydantic>=2.0.0      # Configuration validation
pyyaml>=6.0.0        # YAML parsing
```

### Testing

```bash
# Test imports
python -c "from src import *; print('All imports OK')"

# Test configuration
python -c "from src.config import SystemConfig; c = SystemConfig(); print(c.zmq.address)"

# Test subscriber
python -m src.zmq_subscriber --help
```

### Adding Custom Detection

```python
# src/detection.py
class MyDetectionAlgorithm(DetectionAlgorithm):
    def detect(self, frame: np.ndarray) -> DetectionResult:
        # Your detection logic
        detections = my_model.predict(frame)
        
        return DetectionResult(
            timestamp=time.time(),
            frame_id=self.frame_count,
            detections=detections,
            fps=self.current_fps
        )
```

### Custom Camera Implementation

```python
# src/camera.py
class MyCustomCamera(CameraBase):
    def open(self) -> bool:
        # Initialize camera
        return True
    
    def read(self) -> Optional[np.ndarray]:
        # Capture frame
        return frame
    
    def apply_config(self, config: CameraConfig) -> None:
        # Apply settings
        pass
    
    def close(self) -> None:
        # Cleanup
        pass
```

### Debugging

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check ZMQ connectivity:

```bash
# Terminal 1: Start publisher
python main.py

# Terminal 2: Test subscriber
python -m src.zmq_subscriber --enhanced

# Should see messages arriving
```

---

## Troubleshooting

### Config Not Persisting

✅ **Solution**: Web UI automatically saves on "Apply" - ensure you click the button.

### ZMQ Connection Failed

```bash
# Check if socket file exists (IPC)
ls -l /tmp/rgbtrack.sock

# Check if port is available (TCP)
netstat -tuln | grep 5555

# Test with different transport
python -m src.zmq_subscriber --transport tcp --host localhost --port 5555
```

### Camera Not Opening

✅ **Solution**: Using `DummyCamera` by default - integrate real camera SDK in `src/camera.py`.

### Performance Issues

- **Use IPC transport** for local subscribers (1.5-2x faster)
- Reduce `max_fps` in config
- Lower detection resolution
- Use `high_speed` camera mode

### Port Already in Use

```yaml
# Change UI port in config.yaml
ui_port: 8080  # Instead of 7860
```

---

## See Also

- [config.example.yaml](../config.example.yaml) - Example configuration
- [main.py](../main.py) - Application entry point
- [Pydantic Documentation](https://docs.pydantic.dev/) - Configuration validation
- [ZeroMQ Guide](https://zeromq.org/get-started/) - Messaging patterns

---

**Version**: 1.0  
**Last Updated**: 2024
