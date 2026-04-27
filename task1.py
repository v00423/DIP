# -*- coding: utf-8 -*-
"""
Task 1: Bundle Adjustment with PyTorch
读取风格与仓库 visualize_data.py 保持一致：
- points2d = np.load(".../points2d.npz")
- obs = points2d[key]  # (N,3): x,y,visibility
- pts = obs[:, :2]
- vis = obs[:, 2].astype(bool)
"""

import os
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pytorch3d.transforms import euler_angles_to_matrix


def load_observations_like_visualize_data(data_dir):
    """
    与 visualize_data.py 风格统一的数据读取：
    返回:
      obs_all: (V, N, 2) float32
      vis_all: (V, N) bool
      view_keys: [view_000, ..., view_049]
    """
    points2d = np.load(f"{data_dir}/points2d.npz")  # 与仓库写法一致

    # 与仓库命名一致：view_000 ~ view_049
    # 用排序保证顺序稳定
    view_keys = sorted(list(points2d.keys()))

    obs_list = []
    vis_list = []

    for key in view_keys:
        obs = points2d[key]             # (N,3): [x,y,visibility]
        pts = obs[:, :2].astype(np.float32)
        vis = obs[:, 2].astype(bool)

        obs_list.append(pts)
        vis_list.append(vis)

    obs_all = np.stack(obs_list, axis=0)  # (V,N,2)
    vis_all = np.stack(vis_list, axis=0)  # (V,N)
    return obs_all, vis_all, view_keys


def save_colored_obj(obj_path, points_xyz, colors_rgb):
    os.makedirs(os.path.dirname(obj_path), exist_ok=True)
    with open(obj_path, "w", encoding="utf-8") as f:
        for p, c in zip(points_xyz, colors_rgb):
            x, y, z = p.tolist()
            r, g, b = c.tolist()
            f.write(f"v {x:.6f} {y:.6f} {z:.6f} {r:.6f} {g:.6f} {b:.6f}\n")


class BundleAdjustmentModel(nn.Module):
    def __init__(self, V, N, image_w=1024, image_h=1024, init_fov_deg=60.0, init_depth=2.5, device="cpu"):
        super().__init__()
        self.V = V
        self.N = N
        self.W = float(image_w)
        self.H = float(image_h)
        self.cx = self.W / 2.0
        self.cy = self.H / 2.0

        # 焦距初始化: f = H / (2*tan(fov/2))
        init_f = self.H / (2.0 * math.tan(math.radians(init_fov_deg) / 2.0))
        # 数值稳定：softplus(x)≈x (当x很大时)，所以大于阈值时直接用 init_f
        if init_f > 20:
            f_raw_init = init_f
        else:
            f_raw_init = math.log(math.expm1(init_f))  # 比 log(exp(x)-1) 更稳定

        self.f_raw = nn.Parameter(torch.tensor([f_raw_init], dtype=torch.float32, device=device))

        # 每个视角的欧拉角和位移
        self.euler = nn.Parameter(torch.zeros((V, 3), dtype=torch.float32, device=device))
        t0 = torch.zeros((V, 3), dtype=torch.float32, device=device)
        t0[:, 2] = -init_depth
        self.t = nn.Parameter(t0)

        # 3D点初始化
        p0 = 0.1 * torch.randn((N, 3), dtype=torch.float32, device=device)
        p0[:, 2] += 0.5
        self.points3d = nn.Parameter(p0)

    @property
    def f(self):
        return torch.nn.functional.softplus(self.f_raw)[0] + 1e-6

    def project(self):
        # R: (V,3,3)
        R = euler_angles_to_matrix(self.euler, convention="XYZ")

        # (N,3) -> (V,N,3)
        P = self.points3d.unsqueeze(0).expand(self.V, self.N, 3)

        # [Xc,Yc,Zc] = R @ P + T
        Xc = torch.bmm(P, R.transpose(1, 2)) + self.t.unsqueeze(1)  # (V,N,3)

        X = Xc[..., 0]
        Y = Xc[..., 1]
        Z = Xc[..., 2]
        Z = torch.where(Z.abs() < 1e-8, torch.full_like(Z, 1e-8), Z)

        # README hint 指定投影
        u = -self.f * (X / Z) + self.cx
        v = self.f * (Y / Z) + self.cy
        pred = torch.stack([u, v], dim=-1)  # (V,N,2)
        return pred


def robust_reprojection_loss(pred, gt, vis, delta=3.0):
    # pred/gt: (V,N,2), vis:(V,N)
    err = torch.norm(pred - gt, dim=-1)  # (V,N)
    valid_err = err[vis]
    if valid_err.numel() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    loss = torch.sqrt(valid_err * valid_err + delta * delta).mean() - delta
    return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="E:/VsCodeFiles/DIP_Torch_3D/data")
    parser.add_argument("--image_w", type=int, default=1024)
    parser.add_argument("--image_h", type=int, default=1024)
    parser.add_argument("--iters", type=int, default=2500)
    parser.add_argument("--lr_cam", type=float, default=1e-2)
    parser.add_argument("--lr_pts", type=float, default=5e-3)
    parser.add_argument("--init_fov_deg", type=float, default=60.0)
    parser.add_argument("--init_depth", type=float, default=2.5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_dir", type=str, default="E:/VsCodeFiles/DIP_Torch_3D/results_task1")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[Info] device = {args.device}")

    # 与 visualize_data.py 风格统一读取
    obs_np, vis_np, view_keys = load_observations_like_visualize_data(args.data_dir)
    colors_np = np.load(f"{args.data_dir}/points3d_colors.npy").astype(np.float32)

    V, N, _ = obs_np.shape
    print(f"[Info] views={V}, points={N}")

    obs = torch.from_numpy(obs_np).to(args.device)          # (V,N,2)
    vis = torch.from_numpy(vis_np).to(args.device).bool()   # (V,N)
    colors_np = np.clip(colors_np, 0.0, 1.0)

    model = BundleAdjustmentModel(
        V=V,
        N=N,
        image_w=args.image_w,
        image_h=args.image_h,
        init_fov_deg=args.init_fov_deg,
        init_depth=args.init_depth,
        device=args.device
    ).to(args.device)

    optimizer = torch.optim.Adam(
        [
            {"params": [model.f_raw, model.euler, model.t], "lr": args.lr_cam},
            {"params": [model.points3d], "lr": args.lr_pts},
        ]
    )

    losses = []
    for it in range(1, args.iters + 1):
        optimizer.zero_grad()

        pred = model.project()
        loss_reproj = robust_reprojection_loss(pred, obs, vis, delta=3.0)

        reg_points = 1e-4 * (model.points3d ** 2).mean()
        reg_t = 1e-5 * (model.t ** 2).mean()
        reg_euler = 1e-5 * (model.euler ** 2).mean()

        loss = loss_reproj + reg_points + reg_t + reg_euler
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if it % 100 == 0 or it == 1:
            print(f"[Iter {it:04d}] loss={loss.item():.6f} reproj={loss_reproj.item():.6f} f={model.f.item():.3f}")

    # 保存 loss 曲线
    plt.figure(figsize=(8, 5))
    plt.plot(losses, label="total loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Bundle Adjustment Optimization Loss")
    plt.grid(True)
    plt.legend()
    loss_png = f"{args.out_dir}/loss_curve.png"
    plt.savefig(loss_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Save] {loss_png}")

    # 保存彩色点云
    points3d_final = model.points3d.detach().cpu().numpy()
    obj_path = f"{args.out_dir}/reconstructed_points_colored.obj"
    save_colored_obj(obj_path, points3d_final, colors_np)
    print(f"[Save] {obj_path}")

    # 保存参数
    np.savez(
        f"{args.out_dir}/camera_and_points.npz",
        focal=np.array([model.f.item()], dtype=np.float32),
        euler=model.euler.detach().cpu().numpy(),
        t=model.t.detach().cpu().numpy(),
        points3d=points3d_final,
        view_keys=np.array(view_keys),
    )
    print(f"[Save] {args.out_dir}/camera_and_points.npz")
    print("[Done] Task 1 finished.")


if __name__ == "__main__":
    main()