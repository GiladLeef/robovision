
from ultralytics.models.yolo import classify, detect, obb, pose, segment

from .model import YOLO

__all__ = 'classify', 'segment', 'detect', 'pose', 'obb', 'YOLO'
