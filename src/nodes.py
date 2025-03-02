import json
from typing import List, Optional

import cv2
import numpy as np
import torch
from comfy.utils import ProgressBar

from .meshes import ARAPMeshSystem, HomographyMeshSystem


class LoadTrajectories:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "json_string": ("STRING", {"multiline": True}),
            },
        }

    RETURN_TYPES = ("COORDINATES", "INT", "INT")
    RETURN_NAMES = ("trajectories", "n_anchors", "n_frames")
    DESCRIPTION = "Load a list of trajectories from a JSON string."
    FUNCTION = "load"

    CATEGORY = "MeshPhysics/coordinates"

    def load(self, json_string):
        list_of_lists = json.loads(json_string)
        return (list_of_lists, len(list_of_lists), len(list_of_lists[0]))


class JoinTrajectories:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trajectories_1": ("COORDINATES",),
                "trajectories_2": ("COORDINATES",),
            },
            "optional": {
                "n_frames_1": ("INT", {"default": 0, "min": 0, "step": 1}),
                "n_frames_2": ("INT", {"default": 0, "min": 0, "step": 1}),
            },
        }

    RETURN_TYPES = ("COORDINATES",)
    RETURN_NAMES = ("trajectories",)
    DESCRIPTION = "Crop and join trajectories."
    FUNCTION = "join"

    CATEGORY = "MeshPhysics/coordinates"

    def join(self, trajectories_1, trajectories_2, n_frames_1=0, n_frames_2=0):
        assert len(trajectories_1) == len(trajectories_2), "Trajectories must have the same number of tracks"
        n1 = n_frames_1 or len(trajectories_1[0])
        n2 = n_frames_2 or len(trajectories_2[0])

        coordinates = []
        for track1, track2 in zip(trajectories_1, trajectories_2):
            coordinates.append(track1[:n1] + track2[:n2])

        return (coordinates,)


class SplitKJTrajectoriesLoop:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trajectories": ("COORDINATES",),
                "overlap": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "STRING", "INT")
    RETURN_NAMES = ("coordinates_1", "n_frames_1", "coordinates_2", "n_frames_2")
    DESCRIPTION = "Split a list of looped trajectories into two halves with an overlap."
    FUNCTION = "split"

    CATEGORY = "MeshPhysics/coordinates"

    def split(self, trajectories, overlap):
        mid_point = len(trajectories[0]) // 2

        def tokj(coords):
            return [{"x": c[0], "y": c[1]} for c in coords]

        first_half = [json.dumps(tokj(coord_list[: mid_point + overlap])) for coord_list in trajectories]
        second_half = [json.dumps(tokj(coord_list[mid_point:] + coord_list[:overlap])) for coord_list in trajectories]
        return (first_half, mid_point + overlap, second_half, len(trajectories[0]) - mid_point + overlap)


class ARAPCircleMesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "center_x": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),
                "center_y": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),
                "radii": ("STRING",),
                "points_per_ring": ("STRING",),
            },
            "optional": {
                "trajectories": ("COORDINATES",),
            },
        }

    RETURN_TYPES = ("MESH",)
    RETURN_NAMES = ("mesh",)
    DESCRIPTION = "Generate a concentric circle mesh with additional starting points from a list of trajectories."
    FUNCTION = "generate"

    CATEGORY = "MeshPhysics/simulation"

    def generate(self, center_x, center_y, radii, points_per_ring, trajectories=None):
        radii = map(int, radii.split(","))
        points_per_ring = map(int, points_per_ring.split(","))
        trajectories = trajectories or []
        # Get initial positions of features
        vertices = [coords[0] for coords in trajectories]

        for radius, n_points in zip(radii, points_per_ring):
            for theta in np.linspace(0, 2 * np.pi, n_points, endpoint=False):
                x = center_x + radius * np.cos(theta)
                y = center_y + radius * np.sin(theta)
                vertices.append([x, y])

        mesh = ARAPMeshSystem(vertices, len(trajectories))
        return (mesh,)


class HomographyRectangleMesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 32, "max": 4096, "step": 1}),
                "height": ("INT", {"default": 512, "min": 32, "max": 4096, "step": 1}),
            },
            "optional": {
                "trajectories": ("COORDINATES",),
            },
        }

    RETURN_TYPES = ("MESH",)
    RETURN_NAMES = ("mesh",)
    DESCRIPTION = "Generate a simple grid mesh with linear interpolation."
    FUNCTION = "generate"

    CATEGORY = "MeshPhysics/simulation"

    def generate(self, width, height, trajectories=None):
        trajectories = trajectories or []
        n_anchors = len(trajectories)

        # Get initial positions of features + image corners
        vertices = [coords[0] for coords in trajectories]
        vertices.extend([(0, 0), (width, 0), (0, height), (width, height)])

        mesh = HomographyMeshSystem(vertices, n_anchors)
        return (mesh,)


class AnimateMesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESH",),
                "trajectories": ("COORDINATES",),
                "width": ("INT", {"default": 512, "min": 32, "max": 4096, "step": 1}),
                "height": ("INT", {"default": 512, "min": 32, "max": 4096, "step": 1}),
            },
            "optional": {
                "image": ("IMAGE", {"default": None}),
                "triangles": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    DESCRIPTION = "Simulate the movement of mesh points according to a list of trajectories."
    FUNCTION = "simulate"

    CATEGORY = "MeshPhysics/simulation"

    def simulate(
        self,
        mesh: ARAPMeshSystem | HomographyMeshSystem,
        trajectories: List[List[float]],
        width: int,
        height: int,
        image: Optional[torch.Tensor] = None,  # Expected [B,H,W,C] with values 0-255 RGB
        triangles: bool = True,
    ) -> torch.Tensor:  # Returns [B,H,W,C] with values 0-1 RGB
        n_frames = len(trajectories[0])
        frames = []

        if image is not None:
            image = (image * 255.0).cpu().numpy().astype(np.uint8)[..., ::-1]
            if image.shape[0] == 1:
                image_frames = [image[0]] * n_frames
            elif image.shape[0] == n_frames:
                image_frames = image
            else:
                raise ValueError(f"Image sequence length ({image.shape[0]}) must match trajectory length ({n_frames})")
        else:
            triangles = True

        pbar = ProgressBar(n_frames)
        for frame in range(n_frames):
            anchors = [tr[frame] for tr in trajectories]
            mesh.step(anchors)

            # Start with black frame
            frame_img = np.zeros((height, width, 3), dtype=np.uint8)

            if image is not None:
                for simplex in mesh.faces:
                    src_tri = mesh.initial[simplex].astype(np.float32)
                    dst_tri = mesh.points[simplex].astype(np.float32)

                    warp_mat = cv2.getAffineTransform(src_tri[:3], dst_tri[:3])
                    curr_mask = np.zeros((height, width), dtype=np.uint8)
                    cv2.fillPoly(curr_mask, [dst_tri.astype(np.int32)], 1)

                    # Convert RGB tensor to BGR numpy for cv2
                    warped_piece = cv2.warpAffine(image_frames[frame], warp_mat, (width, height), flags=cv2.INTER_LINEAR)

                    # Apply mask
                    valid_mask = curr_mask > 0
                    frame_img[valid_mask] = warped_piece[valid_mask]

            if triangles:
                # Draw triangles - using BGR colors for cv2
                for simplex in mesh.faces:
                    vertices = mesh.points[simplex].astype(np.int32)
                    cv2.polylines(frame_img, [vertices], True, (0, 255, 0), 1, cv2.LINE_AA)

                feature_mask = np.zeros(len(mesh.points), dtype=bool)
                feature_mask[: len(trajectories)] = True

                # Draw non-feature points (blue in RGB = red in BGR)
                for point in mesh.points[~feature_mask]:
                    cv2.circle(frame_img, tuple(point.astype(int)), 3, (0, 255, 0), -1, cv2.LINE_AA)  # BGR: Blue

                # Draw feature points (red in RGB = blue in BGR)
                for point in mesh.points[feature_mask]:
                    cv2.circle(frame_img, tuple(point.astype(int)), 5, (0, 0, 255), -1, cv2.LINE_AA)  # BGR: Red

            # Convert BGR to RGB before returning
            frame_img = frame_img[..., ::-1].copy()
            # Convert to torch tensor [H,W,C]
            frame_tensor = torch.from_numpy(frame_img).float() / 255.0
            frames.append(frame_tensor)
            pbar.update(1)

        return (torch.stack(frames),)  # Returns [B,H,W,C] with values 0-255 RGB


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "LoadTrajectories": LoadTrajectories,
    "JoinTrajectories": JoinTrajectories,
    "SplitKJTrajectoriesLoop": SplitKJTrajectoriesLoop,
    "ARAPCircleMesh": ARAPCircleMesh,
    "HomographyRectangleMesh": HomographyRectangleMesh,
    "AnimateMesh": AnimateMesh,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadTrajectories": "Load Trajectories",
    "JoinTrajectories": "Join Trajectories",
    "SplitKJTrajectoriesLoop": "Split KJ Trajectories Loop",
    "ARAPCircleMesh": "ARAP Circle Mesh",
    "HomographyRectangleMesh": "Homography Rectangle Mesh",
    "AnimateMesh": "Animate Mesh",
}
