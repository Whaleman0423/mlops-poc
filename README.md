# Hello-World MLOps

本專案演示了一個極簡且可重現的 MLOps (機器學習運維) 工作流程：
1. **訓練模型** ([train.py](./train.py))：訓練邏輯迴歸模型，並將產出儲存至 `artifacts/model.pkl` 與 `artifacts/metrics.json`。
2. **命令列預測** ([run_model.py](./run_model.py))：透過 CLI 輸入 JSON 格式的特徵資料進行單次預測。
3. **API 服務** ([app.py](./app.py))：啟動 Flask API 服務，提供 `/predict` 與 `/health` 端點。
4. **容器化** ([Dockerfile](./Dockerfile))：將 Flask 預測服務打包成 Docker 映像檔。
5. **持續整合 (CI)** ([ci.yaml](./.github/workflows/ci.yaml))：利用 GitHub Actions 自動進行測試與模型訓練，並保留產出成品。


---

## 本地快速開始 (Local Quick Start)

### 1. 建立並啟用虛擬環境
建議使用 Python 3.11 ~ 3.12 版本：
```bash
# 建立虛擬環境
python -m venv .venv

# 啟用虛擬環境 (Windows PowerShell)
.venv\Scripts\Activate.ps1

# 啟用虛擬環境 (macOS / Linux)
source .venv/bin/activate
```

### 2. 安裝依賴套件
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 3. 訓練模型
```bash
python train.py
```
* 執行後會於 `artifacts/` 目錄下產生 `model.pkl` (模型檔) 與 `metrics.json` (評估指標)。

### 4. 透過命令列 (CLI) 進行預測
```bash
python run_model.py --input "[5.1, 3.5, 1.4, 0.2]"
```

### 5. 啟動並測試 API 服務
```bash
python app.py
```
* 預設服務將啟動在 `http://127.0.0.1:5001`
* 使用以下 `curl` 指送測試請求：
```bash
curl -X POST "http://127.0.0.1:5001/predict" \
     -H "Content-Type: application/json" \
     -d '{"features":[5.1,3.5,1.4,0.2]}'
```

---

## 使用 Docker 運行服務

### 1. 建立 Docker 映像檔
```bash
docker build -t hello-mlops .
```

### 2. 啟動 Docker 容器
```bash
docker run -p 5001:5001 hello-mlops
```
* 啟動後，同樣可使用 `http://127.0.0.1:5001/predict` 進行預測測試。
