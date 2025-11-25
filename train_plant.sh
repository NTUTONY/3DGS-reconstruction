#!/bin/bash

# ============================================
# === 3DGS 植物建模自動化腳本 (使用 Nerfstudio) ===
# === (路徑已為 metl204 用戶更新)           ===
# === (修正 ns-process-data 參數錯誤)       ===
# ============================================

# --- 1. 用戶設定 ---
# (路徑已根據您的要求修改)

# 您的照片資料夾 (絕對路徑)
IMAGE_DIR="/home/metl204/my_plant_photos"

# 您所有專案的「根目錄」(您已建立)
PROJECT_BASE_DIR="/home/metl204/my_plant_projects"

# 您「這個」特定植物模型的專案名稱 (將會建立在 PROJECT_BASE_DIR 內)
PROJECT_NAME="plant_3dgs"


# --- 2. 腳本正文 (自動產生路徑，請勿修改) ---

# 確保在正確的 conda 環境中
echo "--- 確保您已在 'nerfstudio' conda 環境中！ ---"


# 腳本將使用的路徑
# COLMAP 處理後的資料存放位置
DATA_DIR="${PROJECT_BASE_DIR}/${PROJECT_NAME}/data"
# ns-train 輸出的根目錄 (nerfstudio 會自動管理子資料夾)
TRAIN_OUTPUT_BASE="${PROJECT_BASE_DIR}"

echo "================================================="
echo "--- 專案總根目錄: ${PROJECT_BASE_DIR}"
echo "--- 目前專案名稱: ${PROJECT_NAME}"
echo "--- (1) 照片來源: ${IMAGE_DIR}"
echo "--- (2) 資料輸出 (COLMAP): ${DATA_DIR}"
echo "--- (3) 訓練輸出 (Model): ${PROJECT_BASE_DIR}/${PROJECT_NAME}/"
echo "================================================="


# --- 步驟 3.1: 處理照片 (運行 COLMAP) ---
echo "--- 開始處理影像 (SfM / COLMAP)... ---"
# ns-process-data 會自動建立所需的 data 資料夾

# !--- 此處為修正點 ---!
# 錯誤的參數是 --image-dir，正確的參數是 --data
ns-process-data images \
    --data $IMAGE_DIR \
    --output-dir $DATA_DIR \
    --no-gpu \
    --verbose
# !--- 修正完畢 ---!


# 檢查 COLMAP 是否成功
if [ ! -f "${DATA_DIR}/transforms.json" ]; then
    echo "--- [錯誤] COLMAP 處理失敗！ ---"
    echo "請檢查您的照片品質或 ${IMAGE_DIR} 路徑。"
    exit 1
fi

echo "--- 影像處理完畢！ ---"


# --- 步驟 3.2: 開始訓練 3DGS ---
echo "--- 開始訓練 3D Gaussian Splatting... ---"

# 您的 RTX 3090 (24GB VRAM) 非常強大，可以全速運行
ns-train splatfacto \
    --data $DATA_DIR \
    --project-name $PROJECT_NAME \
    --experiment-name ${PROJECT_NAME}_gsplat \
    --output-dir $TRAIN_OUTPUT_BASE \
    --viewer.quit-on-train-completion True \
    --pipeline.model.sh-degree 3 \
    --vis wandb
    # --vis tensorboard # 如果您偏好 Tensorboard，請用這行替換
    
echo "--- 訓練完成！ ---"
# 顯示最終模型儲存的確切位置
echo "模型已儲存於 ${TRAIN_OUTPUT_BASE}/${PROJECT_NAME}/${PROJECT_NAME}_gsplat"
