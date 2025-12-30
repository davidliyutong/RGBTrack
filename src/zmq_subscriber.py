"""ZeroMQ subscribers for receiving detection results"""

import argparse
import logging
import pickle
import signal
import sys
import time
from typing import Optional

import zmq

from .config import ZMQConfig

logger = logging.getLogger(__name__)


class ZMQSubscriber:
    """
    Basic ZeroMQ subscriber for receiving detection results.
    """

    def __init__(self, address: str):
        """
        Initialize subscriber.
        
        Args:
            address: ZMQ address (e.g., "ipc:///tmp/rgbtrack.sock" or "tcp://localhost:5555")
        """
        self.address = address
        self.context: Optional[zmq.Context] = None
        self.socket: Optional[zmq.Socket] = None

    def connect(self):
        """Connect to the publisher"""
        try:
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.SUB)
            self.socket.connect(self.address)  # pyright: ignore[reportOptionalMemberAccess]
            self.socket.setsockopt(zmq.SUBSCRIBE, b'')  # pyright: ignore[reportOptionalMemberAccess] # Subscribe to all messages

            logger.info(f"ZMQ subscriber connected to {self.address}")

        except Exception as e:
            logger.error(f"Failed to connect subscriber: {e}")
            raise

    def disconnect(self):
        """Disconnect from the publisher"""
        if self.socket is not None:
            self.socket.close()
            self.socket = None

        if self.context is not None:
            self.context.term()
            self.context = None

        logger.info("ZMQ subscriber disconnected")

    def receive(self, timeout_ms: int = 1000) -> Optional[dict]:
        """
        Receive a detection result.

        Args:
            timeout_ms: Timeout in milliseconds

        Returns:
            Detection result dict or None if timeout
        """
        if self.socket is None:
            return None

        try:
            # Check if data is available
            if self.socket.poll(timeout_ms):
                message = self.socket.recv()
                result = pickle.loads(message)
                return result
            else:
                return None

        except Exception as e:
            logger.error(f"Failed to receive message: {e}")
            return None


class EnhancedSubscriber:
    """
    Enhanced subscriber with statistics, display formatting, and clean shutdown.
    
    Features:
    - Real-time statistics (message count, rate)
    - Formatted display of detection results
    - Signal handling for clean shutdown
    - Performance metrics
    """
    
    def __init__(self, config: ZMQConfig, display_interval: int = 1):
        """
        Initialize enhanced subscriber.
        
        Args:
            config: ZMQ configuration
            display_interval: Display every N messages (0 = all messages)
        """
        self.config = config
        self.subscriber = ZMQSubscriber(config.address)
        self.display_interval = display_interval
        
        # State
        self.running = False
        self.message_count = 0
        self.start_time: Optional[float] = None
        
        # Statistics
        self.total_detections = 0
        self.total_processing_time = 0.0
        
    def run(self):
        """Run the subscriber"""
        self._print_header()
        
        # Connect to publisher
        try:
            self.subscriber.connect()
            print("✓ Connected to publisher")
        except Exception as e:
            print(f"✗ Failed to connect: {e}")
            return
        
        # Setup signal handlers for clean shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.running = True
        self.start_time = time.time()
        print("Waiting for messages... (Press Ctrl+C to stop)")
        print("-" * 70)
        
        # Main receive loop
        while self.running:
            try:
                # Receive message with timeout
                result = self.subscriber.receive(timeout_ms=1000)
                
                if result:
                    self.message_count += 1
                    self._update_statistics(result)
                    
                    # Display based on interval
                    if self.display_interval == 0 or self.message_count % self.display_interval == 0:
                        self._display_result(result)
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error receiving message: {e}")
                time.sleep(0.1)
        
        # Cleanup
        self._shutdown()
    
    def _print_header(self):
        """Print subscriber header"""
        print("=" * 70)
        print("RGBTrack Detection Results Subscriber")
        print("=" * 70)
        print(f"Transport: {self.config.transport}")
        print(f"Address:   {self.config.address}")
        print("-" * 70)
    
    def _display_result(self, result: dict):
        """Display detection result"""
        frame_id = result.get('frame_id', 'N/A')
        timestamp = result.get('timestamp', 0)
        num_detections = len(result.get('detections', []))
        processing_time = result.get('processing_time_ms', 0)
        
        # Calculate receive rate
        elapsed = time.time() - self.start_time if self.start_time else 0
        rate = self.message_count / elapsed if elapsed > 0 else 0
        
        print(f"[{self.message_count:5d}] "
              f"Frame: {frame_id:6} | "
              f"Detections: {num_detections:2d} | "
              f"Process: {processing_time:6.2f}ms | "
              f"Rate: {rate:6.1f} msg/s")
    
    def _update_statistics(self, result: dict):
        """Update statistics from result"""
        self.total_detections += len(result.get('detections', []))
        self.total_processing_time += result.get('processing_time_ms', 0)
    
    def _signal_handler(self, sig, frame):
        """Handle shutdown signals"""
        print("\n\nReceived shutdown signal...")
        self.running = False
    
    def _shutdown(self):
        """Clean shutdown with statistics display"""
        print("\n" + "-" * 70)
        print("Shutting down...")
        
        self.subscriber.disconnect()
        
        # Calculate statistics
        elapsed = time.time() - self.start_time if self.start_time else 0
        avg_rate = self.message_count / elapsed if elapsed > 0 else 0
        avg_detections = self.total_detections / self.message_count if self.message_count > 0 else 0
        avg_processing = self.total_processing_time / self.message_count if self.message_count > 0 else 0
        
        # Display statistics
        print("\n" + "=" * 70)
        print("Statistics")
        print("=" * 70)
        print(f"Messages received:      {self.message_count}")
        print(f"Total detections:       {self.total_detections}")
        print(f"Runtime:                {elapsed:.1f} seconds")
        print(f"Average rate:           {avg_rate:.1f} messages/second")
        print(f"Average detections:     {avg_detections:.1f} per frame")
        print(f"Average processing:     {avg_processing:.2f} ms per frame")
        print("=" * 70)
        print("✓ Subscriber stopped")


def create_subscriber_from_config(config: ZMQConfig, enhanced: bool = True) -> ZMQSubscriber | EnhancedSubscriber:
    """
    Factory function to create subscriber from configuration.
    
    Args:
        config: ZMQ configuration
        enhanced: If True, return EnhancedSubscriber, else basic ZMQSubscriber
        
    Returns:
        Subscriber instance
    """
    if enhanced:
        return EnhancedSubscriber(config)
    else:
        return ZMQSubscriber(config.address)


def main():
    """Command-line interface for the subscriber"""
    parser = argparse.ArgumentParser(
        description="Subscribe to RGBTrack detection results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Unix socket (best performance, local only)
  %(prog)s --transport ipc --socket /tmp/rgbtrack.sock
  
  # TCP (works across network)
  %(prog)s --transport tcp --host localhost --port 5555
  
  # Use default configuration (IPC)
  %(prog)s
  
  # Display every 10 messages (reduce output)
  %(prog)s --display-interval 10
  
  # Use basic subscriber (no formatting)
  %(prog)s --basic
        """
    )
    
    parser.add_argument(
        '--transport', '-t',
        choices=['ipc', 'tcp'],
        default='ipc',
        help='Transport protocol (default: ipc)'
    )
    
    parser.add_argument(
        '--socket', '-s',
        default='/tmp/rgbtrack.sock',
        help='Unix socket path for IPC transport (default: /tmp/rgbtrack.sock)'
    )
    
    parser.add_argument(
        '--host', '-H',
        default='localhost',
        help='Host for TCP transport (default: localhost)'
    )
    
    parser.add_argument(
        '--port', '-p',
        type=int,
        default=5555,
        help='Port for TCP transport (default: 5555)'
    )
    
    parser.add_argument(
        '--display-interval',
        type=int,
        default=1,
        help='Display every N messages (0 = all, default: 1)'
    )
    
    parser.add_argument(
        '--basic',
        action='store_true',
        help='Use basic subscriber without enhanced features'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create configuration
    config = ZMQConfig(
        transport=args.transport,
        socket_path=args.socket,
        host=args.host,
        port=args.port
    )
    
    # Run subscriber
    try:
        if args.basic:
            # Basic subscriber - manual loop
            subscriber = ZMQSubscriber(config.address)
            subscriber.connect()
            
            print(f"Connected to {config.address}")
            print("Receiving messages... (Press Ctrl+C to stop)")
            
            count = 0
            while True:
                result = subscriber.receive(timeout_ms=1000)
                if result:
                    count += 1
                    print(f"[{count}] Frame {result.get('frame_id', 'N/A')}: "
                          f"{len(result.get('detections', []))} detections")
        else:
            # Enhanced subscriber
            subscriber = EnhancedSubscriber(config, display_interval=args.display_interval)
            subscriber.run()
            
    except KeyboardInterrupt:
        print("\n\n✓ Stopped by user")
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
