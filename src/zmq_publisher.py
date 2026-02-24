"""ZeroMQ publisher for detection results"""

import logging
import pickle
import threading
import time
from queue import Queue, Empty
from typing import Optional

import zmq

from .config import ZMQConfig
from .detection import DetectionResult

logger = logging.getLogger(__name__)


class ZMQPublisher:
    """
    ZeroMQ publisher that broadcasts detection results to subscribers.
    """

    def __init__(self, config: ZMQConfig):
        self.config = config
        self.context: Optional[zmq.Context] = None
        self.socket: Optional[zmq.Socket] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._queue: Queue[DetectionResult] = Queue(maxsize=100)
        self._message_count = 0

    def start(self):
        """Start the ZeroMQ publisher"""
        if self._running:
            logger.warning("ZMQ publisher already running")
            return

        try:
            # Initialize ZeroMQ
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.PUB)
            self.socket.bind(self.config.address) # pyright: ignore[reportOptionalMemberAccess]

            transport_type = "Unix socket" if self.config.transport == "ipc" else "TCP"
            logger.info(f"ZMQ publisher started on {self.config.address} (transport: {transport_type})")

            self._running = True
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

        except Exception as e:
            logger.error(f"Failed to start ZMQ publisher: {e}")
            self._cleanup()
            raise

    def stop(self):
        """Stop the ZeroMQ publisher"""
        if not self._running:
            return

        logger.info("Stopping ZMQ publisher...")
        self._running = False

        if self._thread is not None:
            self._thread.join(timeout=2.0)

        self._cleanup()
        logger.info("ZMQ publisher stopped")

    def _cleanup(self):
        """Clean up ZeroMQ resources"""
        if self.socket is not None:
            self.socket.close()
            self.socket = None

        if self.context is not None:
            self.context.term()
            self.context = None

    def publish(self, result: DetectionResult) -> bool:
        """
        Publish a detection result.

        Args:
            result: DetectionResult to publish

        Returns:
            True if queued successfully, False if queue is full
        """
        try:
            self._queue.put_nowait(result)
            return True
        except Exception as e:
            logger.warning(f"Failed to queue result for publishing: {e}")
            return False

    def _run(self):
        """Main publisher loop"""
        logger.info("ZMQ publisher thread started")

        while self._running:
            try:
                # Get result from queue with timeout
                result = self._queue.get(timeout=0.1)

                # Serialize the result
                message = self._serialize_result(result)

                # Publish to all subscribers
                if self.socket is not None:
                    self.socket.send(message)
                    self._message_count += 1

                    if self._message_count % 100 == 0:
                        logger.debug(f"Published {self._message_count} messages")

            except Empty:
                # No data in queue, continue
                continue
            except Exception as e:
                logger.error(f"Error in publisher loop: {e}")
                time.sleep(0.1)

        logger.info("ZMQ publisher thread stopped")

    def _serialize_result(self, result: DetectionResult) -> bytes:
        """
        Serialize detection result for transmission.

        Args:
            result: DetectionResult object

        Returns:
            Serialized bytes
        """
        # Use pickle for Python object serialization
        # For cross-language compatibility, consider using JSON or Protocol Buffers
        try:
            return pickle.dumps(result.to_dict())
        except Exception as e:
            logger.error(f"Failed to serialize result: {e}")
            # Return empty dict as fallback
            return pickle.dumps({})

    def get_stats(self) -> dict:
        """Get publisher statistics"""
        return {
            'running': self._running,
            'transport': self.config.transport,
            'address': self.config.address,
            'messages_published': self._message_count,
            'queue_size': self._queue.qsize()
        }
