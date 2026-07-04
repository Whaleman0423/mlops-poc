"""
Flask Web 應用程式，提供模型預測與健康檢查的 API 服務。
"""
from flask import Flask, request, jsonify
import joblib
import os
from pathlib import Path

# 初始化 Flask 應用程式
app = Flask(__name__)
# 定義模型檔案路徑
MODEL_PATH = Path("artifacts/model.pkl")

# 若模型檔案不存在，自動執行訓練以生成模型
if not MODEL_PATH.exists():
    # 方便起見：模型缺失時進行訓練
    import train as _train
    _train.main()

# 載入機器學習模型
model = joblib.load(MODEL_PATH)

@app.route("/health", methods=["GET"])
def health():
    """
    健康檢查 API，用於確認服務是否正常運作。
    """
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    """
    預測 API，接收 JSON 格式的特徵資料並回傳模型預測結果。
    """
    data = request.get_json()
    # 檢查請求中是否包含特徵欄位 'features'
    if not data or "features" not in data:
        return jsonify({"error": "send JSON with key 'features'"}), 400
    features = data["features"]
    try:
        # 使用模型進行預測並回傳結果
        pred = model.predict([features])
        return jsonify({"prediction": int(pred[0])})
    except Exception as e:
        # 捕捉異常並回傳錯誤訊息
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # 啟動 Flask 伺服器，監聽 port 5001
    app.run(host="0.0.0.0", port=5001)
