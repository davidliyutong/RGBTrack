# RGBTrack Documentation

Welcome to RGBTrack - a multi-threaded Python framework for camera-based detection with real-time configuration and ZeroMQ publishing.

## 📚 Documentation

### [Complete Guide](GUIDE.md)
Comprehensive documentation covering all aspects of RGBTrack:
- Quick start and installation
- Architecture overview
- Configuration system
- ZeroMQ communication
- API reference
- Development guide

## 🚀 Quick Start

```bash
# Start the application
python main.py

# Access Web UI
open http://localhost:7860

# Subscribe to results
python -m src.zmq_subscriber
```

## 📖 Key Topics

### Architecture
- **Multi-threaded design** with separate UI, detection, and ZMQ threads
- **Gradio Web UI** for real-time camera configuration
- **ZeroMQ PUB-SUB** for broadcasting detection results
- See [GUIDE.md#architecture-overview](GUIDE.md#architecture-overview)

### Configuration
- **Pydantic-based** type-safe configuration
- **YAML persistence** for automatic save/load
- **Web UI integration** with live updates
- See [GUIDE.md#configuration](GUIDE.md#configuration)

### ZeroMQ
- **IPC (Unix socket)** for optimal local performance
- **TCP** for network communication
- **Multiple subscribers** supported
- See [GUIDE.md#zeromq-communication](GUIDE.md#zeromq-communication)

## 📁 Project Structure

```
RGBTrack/
├── src/               # Source code
│   ├── app.py         # Main application
│   ├── config.py      # Configuration system
│   ├── camera.py      # Camera interface
│   ├── detection.py   # Detection algorithm
│   ├── webui.py       # Gradio Web UI
│   ├── zmq_publisher.py   # ZMQ publisher
│   └── zmq_subscriber.py  # ZMQ subscribers
├── docs/              # Documentation
│   ├── README.md      # This file
│   └── GUIDE.md       # Complete guide
├── config.yaml        # Runtime configuration
├── config.example.yaml # Example configuration
├── main.py            # Entry point
└── requirements.txt   # Dependencies
```

## 🛠️ Development

### Installation

```bash
# Using UV (recommended)
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Or using pip
pip install -r requirements.txt
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

## 🔧 Configuration

Configuration is stored in `config.yaml` with automatic persistence:

```yaml
camera:
  device_id: 0
  exposure_time_ms: 30
  gain: 1.0
  mode: normal

detection:
  confidence_threshold: 0.5
  mesh_path: models/detector.pt

zmq:
  transport: ipc                # or tcp
  socket_path: /tmp/rgbtrack.sock

ui_host: 0.0.0.0
ui_port: 7860
```

See [GUIDE.md#configuration](GUIDE.md#configuration) for full details.

## 📡 ZeroMQ Usage

### Publisher (automatic in main.py)

The application automatically starts a ZMQ publisher on configured transport.

### Subscriber Examples

```bash
# Standard subscriber (IPC)
python -m src.zmq_subscriber

# Enhanced subscriber with statistics
python -m src.zmq_subscriber --enhanced

# TCP subscriber
python -m src.zmq_subscriber --transport tcp --host localhost --port 5555
```

See [GUIDE.md#zeromq-communication](GUIDE.md#zeromq-communication) for more options.

## 🎯 Common Tasks

### Change Camera Settings
1. Open Web UI at http://localhost:7860
2. Adjust exposure, gain, white balance
3. Click "Apply" - settings auto-save to `config.yaml`

### Switch Between IPC and TCP
```yaml
# config.yaml
zmq:
  transport: ipc  # or tcp
```

### Add Custom Detection
```python
# src/detection.py
class MyDetection(DetectionAlgorithm):
    def detect(self, frame: np.ndarray) -> DetectionResult:
        # Your logic here
        return DetectionResult(...)
```

## 📝 API Reference

See [GUIDE.md#api-reference](GUIDE.md#api-reference) for complete API documentation:
- `RGBTrackApplication` - Main application class
- `CameraBase` - Camera interface
- `DetectionAlgorithm` - Detection base class
- `ZMQPublisher` / `ZMQSubscriber` - ZMQ communication
- `SystemConfig` - Configuration management

## 🐛 Troubleshooting

### Config not persisting
✅ Click "Apply" in Web UI to save changes

### ZMQ connection failed
```bash
# Check socket (IPC)
ls -l /tmp/rgbtrack.sock

# Check port (TCP)
netstat -tuln | grep 5555
```

### Performance issues
- Use IPC transport for local subscribers (faster)
- Reduce `max_fps` in config
- Use `high_speed` camera mode

See [GUIDE.md#troubleshooting](GUIDE.md#troubleshooting) for more solutions.

## 📚 Further Reading

- [Complete Guide](GUIDE.md) - Comprehensive documentation
- [config.example.yaml](../config.example.yaml) - Example configuration
- [main.py](../main.py) - Application entry point

---

**Need help?** Check the [Complete Guide](GUIDE.md) for detailed documentation.

## Other Tasks

### SAM2 Real-Time Checkpoint Model

The SAM2 checkpoint model can be downloaded from the official repository. Please follow these steps:

```
cd segment-anything-2-real-time/checkpoints
bash download_ckpts.sh
```

### RGBTrack Checkpoint Models

The RGBTrack checkpoint models should be put to `weights` folder.

### Build the mycpp module

```shell
cd mycpp
mkdir build
cd build
cmake ..
make -j8
```

### Build cuda plugins

```
cd bundlesdf/mycuda && rm -rf build *egg* *.so
python -m pip install -e .
```