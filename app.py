import os
import csv
import numpy as np
import pickle
import streamlit as st
import shap
import streamlit.components.v1 as components
from pathlib import Path  # 新增




# === 配置路径 ===
BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
MODEL_PATH = BASE_DIR / "models" / "ann_model.pkl"
BACKGROUND_PATH = BASE_DIR / "data" / "shap_background.csv"

# 模型输入的特征顺序（与模型训练/预测一致）
FEATURE_ORDER = ["head", "yinxv", "qiyu", "sds", "sas", "vas"]

# 指定正类标签（如你的正类不是 1，请改成实际标签）
POSITIVE_LABEL = 1

def get_pos_proba(model, X, positive_label=POSITIVE_LABEL) -> np.ndarray:
    """
    返回给定正类标签的概率，按 model.classes_ 映射到正确列。
    兼容无 predict_proba 的模型（如部分 Keras）：对 predict 输出做合理推断。
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        proba = np.asarray(proba)
        if proba.ndim == 1 or proba.shape[1] == 1:
            return np.ravel(proba)
        classes = getattr(model, "classes_", None)
        if classes is not None and positive_label in list(classes):
            idx = list(classes).index(positive_label)
        else:
            # 没有 classes_ 时，二分类默认取最后一列
            idx = proba.shape[1] - 1
        return proba[:, idx]
    # 无 predict_proba：用 predict 做退路（Keras 常见）
    y = model.predict(X)
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] >= 2:
        classes = getattr(model, "classes_", None)
        if classes is not None and positive_label in list(classes):
            idx = list(classes).index(positive_label)
        else:
            idx = y.shape[1] - 1
        return y[:, idx]
    return np.ravel(y).astype(float)

# 读取 background，并按 FEATURE_ORDER 重排
def load_background(path) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        st.error(f"未找到背景数据文件: {p}")
        st.stop()
    rows = []
    with p.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = [h.strip() for h in (reader.fieldnames or [])]
        need = set(FEATURE_ORDER)
        if not need.issubset(set(header)):
            st.error(f"背景数据缺少列。需要: {FEATURE_ORDER}，实际: {header}")
            st.stop()
        for r in reader:
            rows.append([float(r[name]) for name in FEATURE_ORDER])
    X = np.asarray(rows, dtype=float)
    # 背景太大时可下采样以加速（这里保留最多100行）
    if X.shape[0] > 100:
        X = X[:100]
    return X

# 用于在 Streamlit 中渲染 SHAP force_plot（JS 版）
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

# 加载模型
with open(MODEL_PATH, "rb") as f:
    ann_model = pickle.load(f)

# 背景数据（用于 SHAP）
background_X = load_background(BACKGROUND_PATH)

# 根据模型能力，定义用于 SHAP 的预测函数（与页面显示一致）
def model_fn(X: np.ndarray) -> np.ndarray:
    return get_pos_proba(ann_model, X)

# 构建通用 Explainer（未知模型类型时用 Kernel）
explainer = shap.Explainer(model_fn, background_X, feature_names=FEATURE_ORDER)

def main():
    st.title("CRI预测")

    # 数值输入
    sds = st.number_input("SDS抑郁评分", min_value=25.0, max_value=100.0, value=30.0, step=1.0)
    sas = st.number_input("SAS焦虑评分", min_value=25.0, max_value=100.0, value=30.0, step=1.0)
    vas = st.number_input("VAS疼痛评分", min_value=0.0, max_value=10.0, value=1.0, step=1.0)

    # 类别输入（无=0，有=1）
    yinxv = 1 if st.radio("阴虚质", ["无", "有"], index=0, horizontal=True) == "有" else 0
    qiyu  = 1 if st.radio("气郁质", ["无", "有"], index=0, horizontal=True) == "有" else 0
    head  = 1 if st.radio("头颈部肿瘤", ["无", "有"], index=0, horizontal=True) == "有" else 0

    if st.button("预测"):
        # 注意顺序：head, yinxv, qiyu, sds, sas, vas
        X = np.array([[float(head), float(yinxv), float(qiyu),
                       float(sds), float(sas), float(vas)]], dtype=float)

        # 类别/概率输出
        try:
            pred_label = ann_model.predict(X)[0]
            st.subheader("预测结果")
            st.write(f"CRI预测: {pred_label}")
        except Exception as e:
            st.warning(f"预测标签失败：{e}")

        try:
            proba = float(get_pos_proba(ann_model, X)[0])
            st.write(f"发生概率: {proba * 100:.2f}%")
        except Exception as e:
            st.info(f"无法计算概率：{e}")

        # 调试信息（确认送入模型的特征和值）
        with st.expander("调试信息"):
            st.write("classes_:", getattr(ann_model, "classes_", None))
            st.write("送入模型的特征和值（按顺序）:",
                     dict(zip(FEATURE_ORDER, X[0].tolist())))

        # SHAP 解释（Kernel；对单个样本）
        with st.expander("查看SHAP解释（基于背景数据）"):
            try:
                exp = explainer(X)  # Explanation 对象
                base = float(np.ravel(exp.base_values)[0])
                values = np.array(exp.values)[0]  # (n_features,)
                st_shap(
                    shap.force_plot(base, values, X[0], feature_names=FEATURE_ORDER),
                    height=300
                )
            except Exception as e:
                st.error(
                    f"SHAP 解释失败：{e}\n可尝试减少背景样本量，或检查模型输出是否为连续/概率。"
                )

if __name__ == "__main__":
    main()
