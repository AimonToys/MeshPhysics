import json

import cv2
import igl
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay


class ARAPMeshSystem:
    def __init__(self, points, n_anchors):
        self.points = np.array(points, dtype=np.float64)
        # Create triangulation for faces
        self.faces = Delaunay(points).simplices.astype(np.int32)
        # Get initial boundary points (will be updated in step)
        self.b = np.array(range(n_anchors), dtype=np.int32)

        # Initialize ARAP solver
        self.arap_data = igl.ARAP(
            v=self.points.copy(),  # Initial vertex positions
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
    RETURN_NAMES = ("image",)
    DESCRIPTION = "Simulate the movement of mesh points according to a list of trajectories."
    FUNCTION = "simulate"

    CATEGORY = "MeshPhysics/simulation"

    def simulate(self, mesh, trajectories, width, height, image=None, triangles=True):
        # Get number of frames from trajectories
        n_frames = len(trajectories[0])
        frames = []

        # Convert image input to list of frames if needed
        if image is not None:
            if len(image.shape) == 3:  # Single image (H,W,C)
                image_frames = [image] * n_frames
            elif len(image.shape) == 4:  # Sequence of images
                if image.shape[0] == 1:  # Batch 1 (1,H,W,C)
                    image_frames = [image[0]] * n_frames
                elif image.shape[0] == n_frames:
                    image_frames = image
                else:
                    raise ValueError(f"Image sequence length ({image.shape[0]}) must match trajectory length ({n_frames})")
            # Ensure images are float32 in [0,1]
            image_frames = [img.astype(np.float32) if img.dtype != np.float32 else img for img in image_frames]
        else:
            triangles = True

        # Create a single figure and axis for reuse
        if triangles:
            fig, ax = plt.subplots(figsize=(width/100, height/100))

        # Generate each frame
        for frame in range(n_frames):
            # Update mesh positions with current frame's anchors
            anchors = [tr[frame] for tr in trajectories]
            mesh.step(anchors)

            frame_img = np.zeros((height, width, 3), dtype=np.float32)

            # For each triangle in the mesh wrap the image
            if image is not None:
                for simplex in mesh.faces:
                    src_tri = mesh.initial[simplex].astype(np.float32)
                    dst_tri = mesh.points[simplex].astype(np.float32)

                    # Calculate affine transform
                    warp_mat = cv2.getAffineTransform(src_tri[:3], dst_tri[:3])

                    # Create mask for this triangle
                    curr_mask = np.zeros((height, width), dtype=np.uint8)
                    cv2.fillPoly(curr_mask, [dst_tri.astype(np.int32)], 1)

                    # Warp the current frame
                    img_piece = (image_frames[frame] * 255).astype(np.uint8)
                    if img_piece.shape[-1] == 4:
                        img_piece = img_piece[..., :3]
                    warped_piece = cv2.warpAffine(img_piece, warp_mat, (width, height), 
                                                flags=cv2.INTER_LINEAR)

                    # Blend into frame where mask is valid
                    valid_mask = curr_mask > 0
                    frame_img[valid_mask] = warped_piece[valid_mask] / 255.0

            if triangles:
                # Clear the axis instead of creating new figure
                ax.clear()
                # Display current frame
                ax.imshow(frame_img)
                # Draw triangles
                for simplex in mesh.faces:
                    vertices = mesh.points[simplex]
                    ax.fill(vertices[:, 0], vertices[:, 1], 
                           alpha=0.1, color='blue', edgecolor='blue')

                # Draw points
                feature_mask = np.zeros(len(mesh.points), dtype=bool)
                feature_mask[:len(trajectories)] = True  # First n points are features

                ax.scatter(mesh.points[~feature_mask, 0], mesh.points[~feature_mask, 1], 
                          c='b', s=20, alpha=0.5)
                ax.scatter(mesh.points[feature_mask, 0], mesh.points[feature_mask, 1], 
                          c='r', s=50)

                ax.set_xlim(0, width)
                ax.set_ylim(height, 0)

                # Convert plot to image array
                fig.canvas.draw()
                buf = fig.canvas.buffer_rgba()
                # Convert to numpy array and drop alpha channel
                frame_img = np.asarray(buf)[..., :3]
                frame_img = frame_img.astype(np.float32) / 255.0

            frames.append(frame_img)

        # Close the figure after all frames are done
        if triangles:
            plt.close(fig)
        # Stack frames into a single array
        return np.stack(frames)


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

