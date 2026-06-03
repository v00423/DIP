import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from dataclasses import dataclass
import numpy as np
import cv2


class GaussianRenderer(nn.Module):
    def __init__(self, image_height: int, image_width: int):
        super().__init__()
        self.H = image_height
        self.W = image_width

        # Pre-compute pixel coordinates grid
        y, x = torch.meshgrid(
            torch.arange(image_height, dtype=torch.float32),
            torch.arange(image_width, dtype=torch.float32),
            indexing='ij'
        )
        # Shape: (H, W, 2)
        self.register_buffer('pixels', torch.stack([x, y], dim=-1))

    def compute_projection(
        self,
        means3D: torch.Tensor,          # (N, 3)
        covs3d: torch.Tensor,           # (N, 3, 3)
        K: torch.Tensor,                # (3, 3)
        R: torch.Tensor,                # (3, 3)
        t: torch.Tensor                 # (3)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        N = means3D.shape[0]

        # 1. Transform points to camera space
        cam_points = means3D @ R.T + t.unsqueeze(0)  # (N, 3)

        # 2. Get depths before projection for proper sorting and clipping
        depths = cam_points[:, 2].clamp(min=1.0)  # (N,)

        # 3. Project to screen space using camera intrinsics
        screen_points = cam_points @ K.T  # (N, 3)
        means2D = screen_points[..., :2] / screen_points[..., 2:3]  # (N, 2)

        # 4. Transform covariance to camera space and then to 2D
        # Jacobian for u = s0/z, v = s1/z, where:
        # s0 = fx*X + cx*Z, s1 = fy*Y + cy*Z, z = Z
        fx = K[0, 0]
        fy = K[1, 1]
        z = depths  # (N,)
        s0 = screen_points[:, 0]
        s1 = screen_points[:, 1]

        J_proj = torch.zeros((N, 2, 3), device=means3D.device, dtype=means3D.dtype)
        J_proj[:, 0, 0] = fx / z
        J_proj[:, 0, 1] = 0.0
        J_proj[:, 0, 2] = -s0 / (z * z)
        J_proj[:, 1, 0] = 0.0
        J_proj[:, 1, 1] = fy / z
        J_proj[:, 1, 2] = -s1 / (z * z)

        # Apply world-to-camera rotation to the 3D covariance matrix:
        # Sigma_cam = R * Sigma_world * R^T
        covs_cam = torch.bmm(
            R.unsqueeze(0).expand(N, -1, -1),
            torch.bmm(covs3d, R.t().unsqueeze(0).expand(N, -1, -1))
        )  # (N, 3, 3)

        # Project to 2D
        covs2D = torch.bmm(J_proj, torch.bmm(covs_cam, J_proj.permute(0, 2, 1)))  # (N, 2, 2)

        return means2D, covs2D, depths

    def compute_gaussian_values(
        self,
        means2D: torch.Tensor,    # (N, 2)
        covs2D: torch.Tensor,     # (N, 2, 2)
        pixels: torch.Tensor      # (H, W, 2)
    ) -> torch.Tensor:           # (N, H, W)
        N = means2D.shape[0]
        H, W = pixels.shape[:2]

        # Compute offset from mean (N, H, W, 2)
        dx = pixels.unsqueeze(0) - means2D.reshape(N, 1, 1, 2)

        # Add small epsilon to diagonal for numerical stability
        eps = 1e-4
        covs2D = covs2D + eps * torch.eye(2, device=covs2D.device, dtype=covs2D.dtype).unsqueeze(0)

        # Inverse & determinant
        inv_cov = torch.inverse(covs2D)  # (N,2,2)
        det_cov = torch.det(covs2D).clamp_min(1e-12)  # (N,)

        # Quadratic form: (x-mu)^T Sigma^{-1} (x-mu)
        q = torch.einsum('nhwi,nij,nhwj->nhw', dx, inv_cov, dx)  # (N,H,W)

        # Normalized 2D Gaussian
        norm = (1.0 / (2.0 * np.pi * torch.sqrt(det_cov))).view(N, 1, 1)
        gaussian = norm * torch.exp(-0.5 * q)  # (N,H,W)

        return gaussian

    def forward(
        self,
        means3D: torch.Tensor,          # (N, 3)
        covs3d: torch.Tensor,           # (N, 3, 3)
        colors: torch.Tensor,           # (N, 3)
        opacities: torch.Tensor,        # (N, 1)
        K: torch.Tensor,                # (3, 3)
        R: torch.Tensor,                # (3, 3)
        t: torch.Tensor                 # (3, 1)
    ) -> torch.Tensor:
        N = means3D.shape[0]

        # 1. Project to 2D
        means2D, covs2D, depths = self.compute_projection(means3D, covs3d, K, R, t)

        # 2. Depth mask
        valid_mask = (depths > 1.0) & (depths < 50.0)  # (N,)

        # 3. Sort by depth (near -> far)
        indices = torch.argsort(depths, dim=0, descending=False)
        means2D = means2D[indices]
        covs2D = covs2D[indices]
        colors = colors[indices]
        opacities = opacities[indices]
        valid_mask = valid_mask[indices]

        # 4. Compute gaussian values
        gaussian_values = self.compute_gaussian_values(means2D, covs2D, self.pixels)  # (N,H,W)

        # 5. Apply valid mask
        gaussian_values = gaussian_values * valid_mask.view(N, 1, 1)

        # 6. Alpha composition setup
        alphas = opacities.view(N, 1, 1) * gaussian_values  # (N,H,W)
        # Stability clamp (avoid cumprod going negative or to exact 0 too early)
        alphas = alphas.clamp(0.0, 0.999)

        colors = colors.view(N, 3, 1, 1).expand(-1, -1, self.H, self.W)  # (N,3,H,W)
        colors = colors.permute(0, 2, 3, 1)  # (N,H,W,3)

        # 7. Compute weights for front-to-back alpha blending
        # T_i = Π_{j<i} (1 - α_j),  w_i = α_i * T_i
        one_minus_alpha = 1.0 - alphas  # (N,H,W)
        T_inclusive = torch.cumprod(one_minus_alpha + 1e-10, dim=0)  # inclusive product
        T_exclusive = torch.cat([torch.ones_like(T_inclusive[:1]), T_inclusive[:-1]], dim=0)
        weights = alphas * T_exclusive  # (N,H,W)

        # 8. Final rendering
        rendered = (weights.unsqueeze(-1) * colors).sum(dim=0)  # (H,W,3)

        return rendered