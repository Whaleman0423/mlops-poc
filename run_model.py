#!/usr/bin/env python3
"""
使用說明：
    python run_model.py --input "[5.1, 3.5, 1.4, 0.2]"
"""

import argparse
import json
from pathlib import Path
import numpy as np
import joblib

# 定義模型檔案路徑
MODEL_PATH = Path("artifacts/model.pkl")

def load_model():
    """
    載入已訓練的模型。若檔案不存在則拋出 FileNotFoundError。
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"找不到模型檔案：{MODEL_PATH}")
    return joblib.load(MODEL_PATH)   # <-- 正確地載入二進位模型檔案

def main():
    """
    主程式：解析命令列參數、處理輸入特徵、載入模型進行預測並輸出結果。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                        help="JSON 字串格式的特徵列表。範例：\"[5.1,3.5,1.4,0.2]\"")
    args = parser.parse_args()

    # 解析輸入資料
    try:
        features = json.loads(args.input)
    except json.JSONDecodeError:
        raise ValueError("無效的輸入格式。請使用 JSON 列表，例如：--input \"[5.1,3.5,1.4,0.2]\"")

    # 將特徵轉換為 NumPy 二維陣列以符合模型輸入要求
    X = np.array(features).reshape(1, -1)

    # 載入模型並進行預測
    model = load_model()
    pred = model.predict(X)

    # 輸出預測結果（JSON 格式）
    print(json.dumps({"prediction": pred.tolist()}))

if __name__ == "__main__":
    main()

