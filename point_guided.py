import cv2
import numpy as np
import gradio as gr

# Global variables for storing source and target control points
points_src = []
points_dst = []
image = None

# Reset control points when a new image is uploaded
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()
    points_dst.clear()
    image = img
    return img

# Record clicked points and visualize them on the image
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]

    # Alternate clicks between source and target points
    if len(points_src) == len(points_dst):
        points_src.append([x, y])
    else:
        points_dst.append([x, y])

    # Draw points (blue: source, red: target) and arrows on the image
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # Blue for source
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # Red for target

    # Draw arrows from source to target points
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)

    return marked_image

# Point-guided image deformation
def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):#source_pts: 原始控制点 target_pts: 目标控制点
    """
    
     Return
     ------
        A deformed image.
    """
    if len(source_pts) == 0 or len(source_pts) != len(target_pts):
        return np.array(image)  # 没有点或点数不成对匹配，直接返回原图

    # 转为 numpy float32 防止数据类型问题（前期已经经过多次integer数据类型的报错TvT）
    p = np.array(source_pts, dtype=np.float32)   # 源点 (n, 2)
    q = np.array(target_pts, dtype=np.float32)   # 目标点 (n, 2)
    image = np.array(image)                      # 确保是 ndarray

    h, w = image.shape[:2]
    channels = image.shape[2] if image.ndim == 3 else 1

    # 输出图像
    warped_image = np.zeros_like(image)

    # MLS算法函数
    def mls_affine(v, ctrl_p, ctrl_q):
        #计算单个点 v 的 MLS 映射
        n = ctrl_p.shape[0]
        # 权重计算，根据距离计算权重，距离越近权重越大
        dist = np.sum((ctrl_p - v) ** 2, axis=1)
        w = 1.0 / (dist ** alpha + eps)

        sum_w = np.sum(w)
        if sum_w < 1e-6:
            return v.copy()

        # 加权中心
        p_star = np.sum(w[:, None] * ctrl_p, axis=0) / sum_w
        q_star = np.sum(w[:, None] * ctrl_q, axis=0) / sum_w

        # 中心化
        hat_p = ctrl_p - p_star
        hat_q = ctrl_q - q_star

        # 构造矩阵 M (2x2) 和 B (2x2)
        M = np.zeros((2, 2), dtype=np.float32)
        B = np.zeros((2, 2), dtype=np.float32)
        for i in range(n):
            M += w[i] * np.outer(hat_p[i], hat_p[i])
            B += w[i] * np.outer(hat_q[i], hat_p[i])

        # 奇异情况处理（点数少或共线）
        if np.abs(np.linalg.det(M)) < eps:
            return v - p_star + q_star  # 退化为纯平移

        A = B @ np.linalg.inv(M)
        return A @ (v - p_star) + q_star

    # 逐像素backward warping
    for y in range(h):
        for x in range(w):
            v = np.array([x, y], dtype=np.float32)                  # 输出坐标
            # 关键：互换 source 和 target 点的顺序，计算从输出坐标 v 到输入坐标 src_pos 的映射
            src_pos = mls_affine(v, q, p)                           # 互换 q 和 p

            sx, sy = src_pos
            # 边界处理
            sx = np.clip(sx, 0, w - 1)
            sy = np.clip(sy, 0, h - 1)

            # 双线性插值（也可以改为最近邻插值简化实现，双线性清晰度效果更佳）
            x0 = int(np.floor(sx))
            y0 = int(np.floor(sy))
            x1 = min(x0 + 1, w - 1)
            y1 = min(y0 + 1, h - 1)
            dx = sx - x0
            dy = sy - y0

            if channels == 3:
                c00 = image[y0, x0].astype(np.float32)
                c10 = image[y0, x1].astype(np.float32)
                c01 = image[y1, x0].astype(np.float32)
                c11 = image[y1, x1].astype(np.float32)

                top = c00 * (1 - dx) + c10 * dx
                bot = c01 * (1 - dx) + c11 * dx
                val = top * (1 - dy) + bot * dy
                warped_image[y, x] = np.clip(val, 0, 255).astype(np.uint8)
            else:
                # 灰度图处理
                c00 = image[y0, x0]
                c10 = image[y0, x1]
                c01 = image[y1, x0]
                c11 = image[y1, x1]
                val = (c00 * (1 - dx) + c10 * dx) * (1 - dy) + \
                      (c01 * (1 - dx) + c11 * dx) * dy
                warped_image[y, x] = np.clip(val, 0, 255).astype(np.uint8)

    return warped_image

def run_warping():
    global points_src, points_dst, image

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image

# Clear all selected points
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image

# Build Gradio interface
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload Image", interactive=True, width=800)
            point_select = gr.Image(label="Click to Select Source and Target Points", interactive=True, width=800)

        with gr.Column():
            result_image = gr.Image(label="Warped Result", width=800)

    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")

    input_image.upload(upload_image, input_image, point_select)
    point_select.select(record_points, None, point_select)
    run_button.click(run_warping, None, result_image)
    clear_button.click(clear_points, None, point_select)

demo.launch()