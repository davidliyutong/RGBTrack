"""Shared ZeroMQ protocol constants for RGBTrack inference service."""

from typing import Final, FrozenSet

MSG_TYPE_RESULT: Final[str] = "result"
MSG_TYPE_COMMAND: Final[str] = "command"
MSG_TYPE_STATUS: Final[str] = "status"
MSG_TYPE_ERROR: Final[str] = "error"
MSG_TYPES: FrozenSet[str] = frozenset(
	{MSG_TYPE_RESULT, MSG_TYPE_COMMAND, MSG_TYPE_STATUS, MSG_TYPE_ERROR}
)

CMD_START: Final[str] = "start_detection"
CMD_PAUSE: Final[str] = "pause"
CMD_RESUME: Final[str] = "resume"
CMD_RESET: Final[str] = "reset"
CMD_ENABLE_FRAME_BUFFER: Final[str] = "enable_frame_buffer"
CMD_DISABLE_FRAME_BUFFER: Final[str] = "disable_frame_buffer"
CMD_SET_PROMPT: Final[str] = "set_prompt"
CMD_SET_NMS_THRESHOLD: Final[str] = "set_nms_threshold"
CMD_GET_STATUS: Final[str] = "get_status"

COMMAND_TYPES: FrozenSet[str] = frozenset(
	{
		CMD_START,
		CMD_PAUSE,
		CMD_RESUME,
		CMD_RESET,
		CMD_ENABLE_FRAME_BUFFER,
		CMD_DISABLE_FRAME_BUFFER,
		CMD_SET_PROMPT,
		CMD_SET_NMS_THRESHOLD,
		CMD_GET_STATUS,
	}
)

KEY_TYPE: Final[str] = "type"
KEY_COMMAND: Final[str] = "command"
KEY_PAYLOAD: Final[str] = "payload"
KEY_MESSAGE: Final[str] = "message"
KEY_TIMESTAMP: Final[str] = "timestamp"
KEY_SUCCESS: Final[str] = "success"
KEY_STATUS: Final[str] = "status"
KEY_FRAME_BUFFER_ENABLED: Final[str] = "frame_buffer_enabled"
KEY_PROMPT: Final[str] = "prompt"
KEY_NMS_THRESHOLD: Final[str] = "nms_threshold"

STATUS_IDLE: Final[str] = "IDLE"
STATUS_DETECTING: Final[str] = "DETECTING"
STATUS_TRACKING: Final[str] = "TRACKING"
STATUS_PAUSED: Final[str] = "PAUSED"
STATUS_ERROR: Final[str] = "ERROR"

