"""Demo client for RGBTrack ZMQ subscriber.

Usage:
  python demo_zmq_subscriber.py
  python demo_zmq_subscriber.py --auto-start --enable-frame-buffer
  python demo_zmq_subscriber.py --duration 30
"""

from __future__ import annotations

import argparse
import signal
import time
from collections import deque
from pathlib import Path

from src.config import SystemConfig
from src.zmq_subscriber import ZMQSubscriber


class DemoClient:
    """Simple demo client that receives results and computes result-rate FPS."""

    def __init__(self, config: SystemConfig):
        self.config = config
        self.subscriber = ZMQSubscriber(config.zmq)
        self.running = True

        self._arrival_timestamps: deque[float] = deque(maxlen=500)
        self._result_count = 0

    def stop(self) -> None:
        self.running = False
        self.subscriber.disconnect()

    def run(
        self,
        *,
        duration_sec: float = 0.0,
        auto_start: bool = False,
        enable_frame_buffer: bool = False,
    ) -> None:
        if auto_start:
            response = self.subscriber.start_detection()
            print(f"[CMD] start_detection -> {response}")

        if enable_frame_buffer:
            response = self.subscriber.enable_frame_buffer()
            print(f"[CMD] enable_frame_buffer -> {response}")

        print("[INFO] Waiting for results... press Ctrl+C to exit")

        start_time = time.time()
        last_print_ts = 0.0

        while self.running:
            now = time.time()
            if duration_sec > 0 and now - start_time >= duration_sec:
                print("[INFO] Reached duration limit, exiting")
                break

            result_msg = self.subscriber.recv_latest_result(timeout_ms=100)
            if result_msg is not None:
                self._result_count += 1
                self._arrival_timestamps.append(now)

                payload = result_msg.get("payload", {})
                frame_id = payload.get("frame_id", -1)
                inference_fps = payload.get("inference_fps", 0.0)
                camera_fps = payload.get("camera_fps", 0.0)
                has_frame = "frame_jpeg" in payload

                if now - last_print_ts >= 1.0:
                    rx_fps = self._calc_result_rate_fps(now)
                    print(
                        "[RESULT] "
                        f"frame_id={frame_id} "
                        f"rx_fps={rx_fps:.2f} "
                        f"model_infer_fps={float(inference_fps):.2f} "
                        f"camera_fps={float(camera_fps):.2f} "
                        f"frame_buffer={'on' if has_frame else 'off'} "
                        f"total={self._result_count}"
                    )
                    last_print_ts = now
            else:
                if now - last_print_ts >= 2.0:
                    status = self.subscriber.get_status()
                    print(f"[STATUS] {status}")
                    last_print_ts = now

    def _calc_result_rate_fps(self, now: float, window_sec: float = 3.0) -> float:
        while self._arrival_timestamps and now - self._arrival_timestamps[0] > window_sec:
            self._arrival_timestamps.popleft()

        n = len(self._arrival_timestamps)
        if n < 2:
            return 0.0

        dt = self._arrival_timestamps[-1] - self._arrival_timestamps[0]
        if dt <= 1e-6:
            return 0.0
        return (n - 1) / dt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RGBTrack ZMQ subscriber demo client")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Run duration in seconds (0 means run forever)",
    )
    parser.add_argument(
        "--auto-start",
        action="store_true",
        help="Send start_detection command at startup",
    )
    parser.add_argument(
        "--enable-frame-buffer",
        action="store_true",
        help="Enable frame buffer from server at startup",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = SystemConfig.from_yaml(config_path)
    client = DemoClient(config)

    def _handle_signal(signum, frame):
        print(f"\n[INFO] Received signal {signum}, shutting down...")
        client.stop()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        client.run(
            duration_sec=args.duration,
            auto_start=args.auto_start,
            enable_frame_buffer=args.enable_frame_buffer,
        )
    finally:
        client.stop()


if __name__ == "__main__":
    main()
