import cv2
import igl
import numpy as np
from scipy.spatial import Delaunay


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
            f=self.faces,  # Face indices
            dim=2,  # 2D deformation
            b=self.b,  # Boundary vertices
            energy_type=3,  # ARAP_ENERGY_TYPE_SPOKES
            max_iter=5,  # Max iterations for solver
        )

    def step(self, anchors):
        # Update boundary conditions
        bc = np.array(anchors)
        # Use current positions as initial guess
        initial_guess = self.points.copy()
        # Solve for new positions
        self.points = self.arap_data.solve(bc, initial_guess)
        return self.points


class HomographyMeshSystem:
    def __init__(self, points, n_anchors):
        self.points = np.array(points, dtype=np.float64)
        self.initial = self.points.copy()
        self.n_anchors = n_anchors

        # Ensure we have enough anchor points for homography
        assert n_anchors >= 4, "At least 4 anchor points are required for homography transform."

        # Create a simple triangulation for the mesh
        self.faces = Delaunay(points).simplices.astype(np.int32)

    def step(self, anchors):
        # Update points directly from anchors for tracked points
        self.points[: self.n_anchors] = np.array(anchors)
        # Calculate homography matrix from initial to current anchor positions
        H, status = cv2.findHomography(self.initial[: self.n_anchors], self.points[: self.n_anchors], method=cv2.RANSAC)
        if H is not None:
            # Apply homography to non-anchor points
            pts = self.initial[self.n_anchors :].reshape(-1, 1, 2)
            transformed = cv2.perspectiveTransform(pts, H)
            self.points[self.n_anchors :] = transformed.reshape(-1, 2)
        return self.points
