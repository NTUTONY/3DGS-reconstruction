# 🌿 3DGS 植物三維重建與葉面積分析 (Plant Reconstruction & Analysis)

歡迎使用本專案！這是一個專為植物表型分析設計的自動化工具。

我們利用 **Nerfstudio** 與 **3D Gaussian Splatting (3DGS)** 技術，將植物的拍攝影像重建為高精度的 3D 數位模型，並透過程式自動計算出植物的**精確葉面積**。本流程經過簡化，適合初學者與研究人員使用。

---

## 🛠️ 第零步：事前準備 (Prerequisites)

在開始之前，請確認這台電腦已經準備好以下兩樣東西：

1.  **NVIDIA 顯示卡驅動程式**：
    * 本專案需要 NVIDIA 顯示卡 (GPU) 進行運算。請確保驅動程式已更新至最新版本。
2.  **Anaconda (或 Miniconda)**：
    * 這是用來管理程式環境的軟體。
    * 如果你還沒安裝，請至 [Anaconda 官網](https://www.anaconda.com/download) 下載並安裝。
    * *安裝時建議勾選 "Add Anaconda to my PATH environment variable" (Windows 用戶)。*

---

## 📦 第一步：環境安裝與建置 (Installation)

若是**第一次**在這台電腦上使用，請依序執行以下指令來建立環境。
*(請打開終端機 Terminal 或 Anaconda Prompt 執行)*

### 1. 下載專案程式碼
```bash
git clone [https://github.com/NTUTONY/3DGS-reconstruction.git](https://github.com/NTUTONY/3DGS-reconstruction.git)
cd 3DGS-reconstruction
2. 建立虛擬環境
我們要建立一個叫做 nerfstudio 的專屬環境，並安裝 Python 3.8：

Bash

conda create --name nerfstudio python=3.8 -y
3. 啟動環境並安裝套件
環境建立好後，我們需要啟動它，並安裝此專案需要的 Nerfstudio 與其他計算工具：

Bash

# 啟動環境
conda activate nerfstudio

# 安裝核心依賴 (Nerfstudio 及相關套件)
# 注意：這步會下載較多檔案，請保持網路暢通
pip install nerfstudio

# 安裝數據分析需要的額外套件 (如計算葉面積用)
pip install numpy pandas open3d scipy
(如果專案中有提供 requirements.txt，也可以用 pip install -r requirements.txt 代替上方指令)

🚀 第二步：開始訓練模型 (Training)
當環境安裝好後，以後每次使用只需要從這裡開始。

1. 啟動環境
確保你是在 nerfstudio 的環境下操作（終端機最前面會顯示 (nerfstudio)）：

Bash

conda activate nerfstudio
2. 執行自動化訓練腳本
我們將繁雜的指令封裝成了 .sh 腳本。這個腳本會自動讀取數據並訓練 3D 模型。

Bash

# 賦予腳本執行權限 (如果是剛下載的檔案，務必執行這行，否則會報錯)
chmod +x train_plant.sh

# 開始訓練
./train_plant.sh
☕ 小提醒：訓練過程依據照片數量與顯卡效能，可能需要 10~30 分鐘。請耐心等待直到出現 "Finished" 或進度條跑完。

📊 第三步：獲取數據分析 (Analysis)
當 3D 模型訓練完成後，我們執行 Python 程式來提取數據。

Bash

python3 precise_leaf_area.py
程式輸出說明：

程式會讀取剛剛訓練好的模型。

計算並在螢幕上顯示植物的總葉面積 (Leaf Area) 數據。

相關的分析圖表或數據檔將會儲存在專案資料夾中。

📂 檔案結構說明
train_plant.sh：自動化訓練腳本。它是給系統看的「劇本」，告訴電腦如何呼叫 Nerfstudio 進行運算。

precise_leaf_area.py：葉面積計算程式。利用演算法分析 3D 點雲或投影，算出精確數值。

data/：(預設) 請將你拍攝的植物照片或影片放入此資料夾。
