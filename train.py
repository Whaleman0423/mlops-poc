"""
簡單的機器學習模型訓練腳本：
- 從 sklearn 載入鳶尾花 (Iris) 資料集
- 訓練一個邏輯迴歸 (LogisticRegression) 分類模型
- 將訓練好的模型儲存至 artifacts/model.pkl
- 計算並儲存模型的評估指標 (accuracy) 至 artifacts/metrics.json
"""

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import os
import json

def main():
    """
    主程式：載入資料、分割訓練集與測試集、訓練模型、評估並儲存模型與指標。
    """
    # 載入鳶尾花資料集
    iris = load_iris()
    X, y = iris.data, iris.target
    # 分割資料集為訓練集與測試集（比例為 80:20，並設定固定隨機種子以確保結果可重現）
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 初始化邏輯迴歸模型，並設定最大反覆次數為 200
    model = LogisticRegression(max_iter=200)
    # 開始訓練模型
    model.fit(X_train, y_train)

    # 建立模型輸出目錄 (artifacts)
    os.makedirs("artifacts", exist_ok=True)
    model_path = os.path.join("artifacts", "model.pkl")
    # 將訓練完成的模型序列化並儲存
    joblib.dump(model, model_path)

    # 計算模型在測試集上的準確度，並儲存為指標檔案
    acc = model.score(X_test, y_test)
    metrics = {"accuracy": float(acc)}
    with open(os.path.join("artifacts", "metrics.json"), "w") as f:
        json.dump(metrics, f)

    # 印出執行結果資訊
    print(f"已儲存模型至 {model_path}")
    print(f"測試集準確度 (Test accuracy): {acc:.4f}")

if __name__ == "__main__":
    main()
