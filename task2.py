import subprocess
from pathlib import Path
import shutil
import sys

COLMAP_ROOT = Path(r"E:\colmap\colmap-x64-windows-cuda")

def get_colmap_executable():
    candidates = [
        COLMAP_ROOT / "COLMAP.bat",
        COLMAP_ROOT / "colmap.bat",
        COLMAP_ROOT / "COLMAP.exe",
        COLMAP_ROOT / "colmap.exe",
        COLMAP_ROOT / "bin" / "COLMAP.bat",
        COLMAP_ROOT / "bin" / "colmap.bat",
        COLMAP_ROOT / "bin" / "COLMAP.exe",
        COLMAP_ROOT / "bin" / "colmap.exe",
    ]
    for c in candidates:
        if c.exists():
            return [str(c)]
    if shutil.which("colmap"):
        return ["colmap"]

    raise FileNotFoundError(
        f" colmap读取失败 "
    )

def run_cmd(cmd):
    print(">>", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)

def main():
    root = Path(__file__).resolve().parent

    dataset_path = root / "data"
    image_path = dataset_path / "images"
    colmap_path = dataset_path / "colmap"

    sparse_dir = colmap_path / "sparse"
    dense_dir = colmap_path / "dense"
    db_path = colmap_path / "database.db"

    sparse_dir.mkdir(parents=True, exist_ok=True)
    dense_dir.mkdir(parents=True, exist_ok=True)

    colmap = get_colmap_executable()

    print("=== Step 1: Feature Extraction ===")
    run_cmd(colmap + [
        "feature_extractor",
        "--database_path", str(db_path),
        "--image_path", str(image_path),
        "--ImageReader.camera_model", "PINHOLE",
        "--ImageReader.single_camera", "1",
    ])

    print("=== Step 2: Feature Matching ===")
    run_cmd(colmap + [
        "exhaustive_matcher",
        "--database_path", str(db_path),
    ])

    print("=== Step 3: Sparse Reconstruction (Bundle Adjustment) ===")
    run_cmd(colmap + [
        "mapper",
        "--database_path", str(db_path),
        "--image_path", str(image_path),
        "--output_path", str(sparse_dir),
    ])

    print("=== Step 4: Image Undistortion ===")
    run_cmd(colmap + [
        "image_undistorter",
        "--image_path", str(image_path),
        "--input_path", str(sparse_dir / "0"),
        "--output_path", str(dense_dir),
    ])

    print("=== Step 5: Dense Reconstruction (Patch Match Stereo) ===")
    run_cmd(colmap + [
        "patch_match_stereo",
        "--workspace_path", str(dense_dir),
    ])

    print("=== Step 6: Stereo Fusion ===")
    run_cmd(colmap + [
        "stereo_fusion",
        "--workspace_path", str(dense_dir),
        "--output_path", str(dense_dir / "fused.ply"),
    ])

    # 把稀疏模型转成 ply 格式
    print("=== Optional: Convert Sparse Model to PLY ===")
    run_cmd(colmap + [
        "model_converter",
        "--input_path", str(sparse_dir / "0"),
        "--output_path", str(sparse_dir / "0" / "sparse.ply"),
        "--output_type", "PLY",
    ])

    print("=== Done! ===")
    print("Results:")
    print(f"  Sparse(bin): {sparse_dir / '0'}")
    print(f"  Sparse(ply): {sparse_dir / '0' / 'sparse.ply'}")
    print(f"  Dense:       {dense_dir / 'fused.ply'}")

if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] 脚本没跑起来")
        sys.exit(e.returncode)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
