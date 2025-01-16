import json
from typing import List, Optional

import cv2
import igl
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial import Delaunay
from comfy.utils import ProgressBar


class ARAPMeshSystem:
    def __init__(self, points, n_anchors):
        self.points = np.array(points, dtype=np.float64)
        self.initial = self.points.copy()
        # Create triangulation for faces
        self.faces = Delaunay(points).simplices.astype(np.int32)
        # Get initial boundary points (will be updated in step)
        self.b = np.array(range(n_anchors), dtype=np.int32)

        # Initialize ARAP solver
        self.arap_data = igl.ARAP(
            v=self.initial,  # Initial vertex positions
            f=self.faces,          # Face indices
            dim=2,                 # 2D deformation
            b=self.b,             # Boundary vertices
            energy_type=3,        # ARAP_ENERGY_TYPE_SPOKES
            max_iter=5,           # Max iterations for solver
        )

    def step(self, anchors):
        # Update boundary conditions
        bc = np.array(anchors)
        # Use current positions as initial guess
        initial_guess = self.points.copy()
        # Solve for new positions
        self.points = self.arap_data.solve(bc, initial_guess)
        return self.points


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
            return [{'x': c[0], 'y': c[1]} for c in coords]

        first_half = [
            json.dumps(tokj(coord_list[:mid_point + overlap]))
            for coord_list in trajectories
        ]
        second_half = [
            json.dumps(tokj(coord_list[mid_point:] + coord_list[:overlap]))
            for coord_list in trajectories
        ]
        return (first_half, len(first_half[0]), second_half, len(second_half[0]))


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
            for theta in np.linspace(0, 2*np.pi, n_points, endpoint=False):
                x = center_x + radius * np.cos(theta)
                y = center_y + radius * np.sin(theta)
                vertices.append([x, y])

        mesh = ARAPMeshSystem(vertices, len(trajectories))
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
        mesh: ARAPMeshSystem,
        trajectories: List[List[float]], 
        width: int, 
        height: int, 
        image: Optional[torch.Tensor] = None,  # Expected [B,H,W,C] with values 0-255 RGB
        triangles: bool = True
    ) -> torch.Tensor:  # Returns [B,H,W,C] with values 0-1 RGB
        n_frames = len(trajectories[0])
        frames = []

        if image is not None:
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
                    img_piece = image_frames[frame].cpu().numpy().astype(np.uint8)[..., ::-1]
                    warped_piece = cv2.warpAffine(img_piece, warp_mat, (width, height), 
                                                flags=cv2.INTER_LINEAR)

                    # Apply mask
                    valid_mask = curr_mask > 0
                    frame_img[valid_mask] = warped_piece[valid_mask]

            if triangles:
                # Draw triangles - using BGR colors for cv2
                for simplex in mesh.faces:
                    vertices = mesh.points[simplex].astype(np.int32)
                    cv2.polylines(frame_img, [vertices], True, (0, 255, 0), 1, cv2.LINE_AA)

                feature_mask = np.zeros(len(mesh.points), dtype=bool)
                feature_mask[:len(trajectories)] = True

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

        return torch.stack(frames)  # Returns [B,H,W,C] with values 0-255 RGB


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "LoadTrajectories": LoadTrajectories,
    "SplitKJTrajectoriesLoop": SplitKJTrajectoriesLoop,
    "ARAPCircleMesh": ARAPCircleMesh,
    "AnimateMesh": AnimateMesh,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadTrajectories": "Load Trajectories",
    "SplitKJTrajectoriesLoop": "Split KJ Trajectories Loop",
    "ARAPCircleMesh": "ARAP Circle Mesh",
    "AnimateMesh": "Animate Mesh",
}

