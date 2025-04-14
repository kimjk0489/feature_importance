import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score
import shap

# --- 페이지 설정 ---
st.set_page_config(page_title="Slurry 조성 중요도 분석", layout="wide")
st.title("Slurry 조성 중요도 분석 (Random Forest 기반)")

# --- 데이터 불러오기 ---
CSV_PATH = "slurry_data_wt%_ALL.csv"
df = pd.read_csv(CSV_PATH)

# --- 전처리 ---
x_cols = ["carbon_black_wt%", "graphite_wt%", "CMC_wt%", "solvent_wt%"]
y_cols = ["yield_stress"]
X_raw = df[x_cols].values
Y_raw = df[y_cols].values

# 데이터 기반 정규화
x_scaler = MinMaxScaler()
x_scaler.fit(X_raw)
X_scaled = x_scaler.transform(X_raw)

# --- Random Forest 학습 ---
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_scaled, Y_raw.ravel())

# --- Feature Importance 해석 ---
st.subheader("Random Forest 기반 조성 해석")

# Gini Importance
st.markdown("### Gini Importance (트리 분할 기준 중요도)")
gini_importances = rf_model.feature_importances_
sorted_idx = np.argsort(gini_importances)
fig_gini, ax_gini = plt.subplots()
ax_gini.barh(np.array(x_cols)[sorted_idx], gini_importances[sorted_idx], color="skyblue")
ax_gini.set_xlabel("Importance")
ax_gini.set_title("Gini Feature Importance (Random Forest)")
st.pyplot(fig_gini)

# SHAP Importance
st.markdown("### SHAP Importance (예측 기여도 기반 중요도)")
explainer = shap.Explainer(rf_model, X_scaled)
shap_values = explainer(X_scaled, check_additivity=False)
fig_shap = plt.figure()
shap.summary_plot(shap_values, X_scaled, feature_names=x_cols, show=False)
st.pyplot(fig_shap)

# Permutation Importance
st.markdown("### Permutation Importance (성능 변화 기반 중요도)")
perm_result = permutation_importance(rf_model, X_scaled, Y_raw.ravel(), n_repeats=10, random_state=42)
perm_importances = perm_result.importances_mean
perm_std = perm_result.importances_std
sorted_idx = np.argsort(perm_importances)
fig_perm, ax_perm = plt.subplots()
ax_perm.barh(np.array(x_cols)[sorted_idx], perm_importances[sorted_idx], xerr=perm_std[sorted_idx], color="orange")
ax_perm.set_xlabel("Importance")
ax_perm.set_title("Permutation Feature Importance")
st.pyplot(fig_perm)

# Drop-column Importance
st.markdown("### Drop-column Importance (변수 제거 후 재학습 기반 중요도)")
baseline_score = cross_val_score(rf_model, X_scaled, Y_raw.ravel(), cv=5).mean()
drop_importances = {}
for col in x_cols:
    X_drop = pd.DataFrame(X_scaled, columns=x_cols).drop(columns=[col])
    drop_model = RandomForestRegressor(n_estimators=100, random_state=42)
    score = cross_val_score(drop_model, X_drop, Y_raw.ravel(), cv=5).mean()
    drop_importances[col] = baseline_score - score

fig_drop, ax_drop = plt.subplots()
drop_sorted = sorted(drop_importances.items(), key=lambda x: x[1])
labels, values = zip(*drop_sorted)
ax_drop.barh(labels, values, color="purple")
ax_drop.set_xlabel("Importance")
ax_drop.set_title("Drop-column Feature Importance")
st.pyplot(fig_drop)
