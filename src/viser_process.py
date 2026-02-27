"""Viser process wrapper for RGBTrack inference UI."""

from __future__ import annotations

import logging
import multiprocessing as mp
import time
from typing import Optional

from .config import SystemConfig

logger = logging.getLogger(__name__)


class ViserProcess(mp.Process):
    """Separate process launcher for Viser web UI."""

    def __init__(self, config: SystemConfig):
        super().__init__(daemon=True)
        self.config = config

    def run(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        logger.info("Viser process starting")

        try:
            from .viser_ui import start_viser_interface
        except Exception as exc:
            logger.warning("Viser UI entry unavailable: %s", exc)
            while True:
                time.sleep(1.0)

        try:
            start_viser_interface(self.config)
        except KeyboardInterrupt:
            logger.info("Viser process interrupted")
        except Exception as exc:
            logger.error("Viser process failed: %s", exc, exc_info=True)
            while True:
                time.sleep(1.0)
