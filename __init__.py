"""Top-level package for MeshPhysics."""

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]

__author__ = """ComfyUI Mesh Physics"""
__email__ = "matasoff@aimon.toys"
__version__ = "0.0.1"

from .src.MeshPhysics.nodes import NODE_CLASS_MAPPINGS
from .src.MeshPhysics.nodes import NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"
