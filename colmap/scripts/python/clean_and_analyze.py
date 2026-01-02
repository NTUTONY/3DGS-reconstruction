import open3d as o3d
import numpy as np
from plyfile import PlyData
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt  # 新增繪圖庫
import os
import glob

# ==========================================
# 🎛️ 參數設定區 
# ==========================================
# --- 1. 顏色與亮度過濾 ---
GREEN_RATIO_THRESHOLD = 1.05
MIN_BRIGHTNESS_THRESHOLD = 0.2

# --- 2. 空間去噪 (初步過濾) ---
NB_NEIGHBORS = 50
STD_RATIO = 1.0

# --- 3. 主體分離 (DBSCAN 分群 - V3核心) ---
DBSCAN_EPS = 0.02
DBSCAN_MIN_SAMPLES = 10

# --- 4. 葉面積計算 ---
GRID_RESOLUTION = 160

# --- 5. 葉數估計 ---
CLUSTER_EPS_CM = 1.5
MIN_POINTS_IN_CLUSTER = 20

# --- [新增] 6. 角度與萎凋分析 ---
DROOPING_THRESHOLD_DEG = 60.0  # 超過60度視為下垂/直立
# ==========================================

class PlantAnalyzerAngle:
    def __init__(self, ply_path):
        self.ply_path = ply_path
        print(f"\n📂 正在讀取 Splat 檔案: {ply_path} ...")

        plydata = PlyData.read(ply_path)
        vertex = plydata['vertex']

        self.points = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=-1)

        # 讀取顏色 (相容 3DGS f_dc 與一般 RGB)
        SH_C0 = 0.28209479177387814
        try:
            r = 0.5 + SH_C0 * vertex['f_dc_0']
            g = 0.5 + SH_C0 * vertex['f_dc_1']
            b = 0.5 + SH_C0 * vertex['f_dc_2']
            self.colors = np.clip(np.stack([r, g, b], axis=-1), 0, 1)
            print("✅ 成功解碼 3DGS 顏色格式 (f_dc)！")
        except:
            print("⚠️ 找不到 f_dc，嘗試讀取標準 RGB...")
            try:
                r = vertex['red'] / 255.0
                g = vertex['green'] / 255.0
                b = vertex['blue'] / 255.0
                self.colors = np.stack([r, g, b], axis=-1)
            except:
                self.colors = np.zeros_like(self.points) + 0.5

        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(self.points)
        self.pcd.colors = o3d.utility.Vector3dVector(self.colors)
        print(f"   原始點數: {len(self.points)}")

    def clean_background(self):
        """三重過濾：顏色亮度 + 空間去噪 + 保留最大叢集"""
        print("\n🧹 [前處理] 啟動清洗程序...")

        # A. 顏色與亮度
        R, G, B = self.colors[:, 0], self.colors[:, 1], self.colors[:, 2]
        brightness = (R + G + B) / 3.0
        mask_combined = (G > R * GREEN_RATIO_THRESHOLD) & \
                        (G > B * GREEN_RATIO_THRESHOLD) & \
                        (brightness > MIN_BRIGHTNESS_THRESHOLD)
        pcd_green = self.pcd.select_by_index(np.where(mask_combined)[0])

        if len(pcd_green.points) == 0: return None

        # B. 空間去噪
        pcd_sor, _ = pcd_green.remove_statistical_outlier(
            nb_neighbors=NB_NEIGHBORS, std_ratio=STD_RATIO
        )
        if len(pcd_sor.points) == 0: return None

        # C. DBSCAN 主體分離
        print(f"   [處理] 執行主體分離 (保留最大叢集)...")
        points_sor = np.asarray(pcd_sor.points)
        db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(points_sor)
        labels = db.labels_
        unique_labels = set(labels) - {-1}

        if not unique_labels:
            print("❌ 錯誤：無法找到任何叢集，請調整 DBSCAN_EPS。")
            return None

        max_cluster_label = max(unique_labels, key=lambda l: np.sum(labels == l))
        pcd_clean = pcd_sor.select_by_index(np.where(labels == max_cluster_label)[0])

        self.pcd_clean = pcd_clean
        self.points_clean = np.asarray(pcd_clean.points)
        self.colors_clean = np.asarray(pcd_clean.colors)

        print(f"✨ 清洗完成，最終分析點數: {len(self.points_clean)}")
        
        # 存檔
        base_name = os.path.splitext(self.ply_path)[0]
        o3d.io.write_point_cloud(f"{base_name}_cleaned.ply", pcd_clean)
        return pcd_clean

    def get_scale_factor(self):
        print("\n📏 [比例尺校正]")
        try:
            real_len = float(input("   1. 請輸入參照物真實長度 (cm): "))
            model_len = float(input("   2. 請輸入 CloudCompare 量測數值: "))
            scale = real_len / model_len
            print(f"   ✅ Scale Factor: {scale:.4f}")
            return scale
        except:
            return 1.0

    def analyze_metrics(self, scale):
        if not hasattr(self, 'pcd_clean'): return
        print("\n📊 [分析] 正在計算各項指標...")

        points = self.points_clean
        colors = self.colors_clean

        # --- 1. [投影葉面積] ---
        pca = PCA(n_components=3).fit(points)
        points_rot = pca.transform(points) * scale
        points_2d = points_rot[:, :2]
        
        # Grid Calculation
        p_min, p_max = np.min(points_2d, axis=0), np.max(points_2d, axis=0)
        longest = max(p_max - p_min)
        bin_size = longest / GRID_RESOLUTION
        bins = [int(np.ceil((p_max[i]-p_min[i])/bin_size))+1 for i in range(2)]
        H, _, _ = np.histogram2d(points_2d[:, 0], points_2d[:, 1], bins=bins)
        leaf_area = np.sum(H > 0) * (bin_size ** 2)

        # --- 2. [葉節位數] ---
        pcd_temp = o3d.geometry.PointCloud()
        pcd_temp.points = o3d.utility.Vector3dVector(points * scale)
        pcd_down = pcd_temp.voxel_down_sample(voxel_size=0.3)
        pts_down = np.asarray(pcd_down.points)
        
        node_count = 0
        if len(pts_down) > 0:
            db = DBSCAN(eps=CLUSTER_EPS_CM, min_samples=MIN_POINTS_IN_CLUSTER).fit(pts_down)
            node_count = len(set(db.labels_) - {-1})

        # --- 3. [健康度] ---
        avg_r, avg_g, avg_b = np.mean(colors, axis=0)
        raw_exg = (2 * avg_g) - avg_r - avg_b
        health_score = max(0, min(100, (raw_exg / 0.5) * 100))

        # =========================================================
        # 🆕 [新增] 4. 葉片角度分析 (LAD)
        # =========================================================
        print("   📐 正在估算法向量與葉傾角...")
        # A. 估計法向量 (radius=0.03 約為 3cm 鄰域，視模型大小而定)
        self.pcd_clean.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=30))
        
        # B. 統一方向朝上 (Z軸)
        self.pcd_clean.orient_normals_to_align_with_direction(orientation_reference=np.array([0., 0., 1.]))
        normals = np.asarray(self.pcd_clean.normals)
        
        # C. 計算夾角 (0度=平, 90度=垂)
        nz = np.abs(normals[:, 2]) # 取絕對值
        nz = np.clip(nz, -1.0, 1.0)
        angles_deg = np.degrees(np.arccos(nz))
        
        avg_angle = np.mean(angles_deg)
        
        # D. 萎凋指數 (Drooping Index)
        drooping_ratio = np.sum(angles_deg > DROOPING_THRESHOLD_DEG) / len(angles_deg) * 100.0
        
        # E. 視覺化存檔 (顏色映射: 藍=平 -> 紅=垂)
        cmap = plt.get_cmap("coolwarm")
        angle_colors = cmap(angles_deg / 90.0)[:, :3]
        
        viz_pcd = o3d.geometry.PointCloud()
        viz_pcd.points = self.pcd_clean.points
        viz_pcd.colors = o3d.utility.Vector3dVector(angle_colors)
        o3d.io.write_point_cloud("viz_leaf_angles.ply", viz_pcd)
        
        # F. 產生統計圖表
        plt.figure(figsize=(8, 4))
        plt.hist(angles_deg, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        plt.axvline(avg_angle, color='red', linestyle='dashed', label=f'Avg: {avg_angle:.1f}')
        plt.axvline(DROOPING_THRESHOLD_DEG, color='orange', linestyle='dotted', label='Droop Thresh')
        plt.title("Leaf Angle Distribution (LAD)")
        plt.xlabel("Angle (Degree)")
        plt.ylabel("Count")
        plt.legend()
        plt.savefig("chart_leaf_angles.png")

        # --- 輸出總結報告 ---
        print("\n" + "="*45)
        print("📊 植物表型分析總結 (含角度分析)")
        print("="*45)
        print(f"1️⃣  [投影葉面積 (PLA)]")
        print(f"    數值: {leaf_area:.2f} cm²")
        print("-" * 45)
        print(f"2️⃣  [估計葉簇數]")
        print(f"    數值: {node_count} 個")
        print("-" * 45)
        print(f"3️⃣  [生理健康度]")
        print(f"    ExG 指標: {raw_exg:.3f} (評分: {health_score:.1f})")
        print("-" * 45)
        print(f"4️⃣  [姿態與結構 (Posture)]")
        print(f"    平均葉傾角 (MLA) : {avg_angle:.2f}°")
        print(f"    萎凋/下垂指數    : {drooping_ratio:.1f}%")
        print(f"    (視覺化檔案)     : viz_leaf_angles.ply")
        print(f"    (圖表檔案)       : chart_leaf_angles.png")
        print("="*45)

def get_latest_ply():
    files = glob.glob("*.ply")
    # 排除產生的結果檔
    valid_files = [f for f in files if "cleaned" not in f and "viz" not in f and "sparse" not in f]
    if not valid_files: return None
    return max(valid_files, key=os.path.getmtime)

if __name__ == "__main__":
    INPUT_PLY = get_latest_ply()
    if INPUT_PLY:
        print(f"✨ 發現最新模型檔案: {INPUT_PLY}")
        analyzer = PlantAnalyzerAngle(INPUT_PLY)
        if analyzer.clean_background():
            scale_factor = analyzer.get_scale_factor()
            analyzer.analyze_metrics(scale_factor)
    else:
        print("❌ 找不到可處理的 .ply 檔案。")