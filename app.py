import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta, date
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

st.set_page_config(page_title="智慧營運與機器學習預測系統", layout="wide")

st.title("📈 智慧機器學習與情境預測系統")
st.write("請依照下方明確的步驟區塊，上傳 Excel 檔案並選擇工作表進行情境模擬與政策對比預測。")

# ==================== 快取讀取函式 (解決點擊卡頓問題) ====================
@st.cache_data
def load_excel_data(file, sheet):
    df = pd.read_excel(file, sheet_name=sheet)
    pseudo_numeric_keywords = ['yyy', 'ym', '年月', '代號', '年份', '月份']
    for col in df.columns:
        if any(kw in str(col).lower() for kw in pseudo_numeric_keywords):
            df[col] = df[col].astype(str)
    return df

# ==================== 步驟一：資料上傳與工作表選擇 ====================
st.markdown("---")
st.markdown("### 📂 步驟一：上傳資料與工作表選擇")

with st.container(border=True):
    uploaded_file = st.file_uploader("請上傳 Excel 資料檔案 (.xlsx)", type=["xlsx"])
    
    if uploaded_file is not None:
        excel_file = pd.ExcelFile(uploaded_file)
        sheet_names = excel_file.sheet_names
        
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            selected_sheet = st.selectbox("📌 請選擇要讀取的工作表 (Sheet)", sheet_names)
        with col_s2:
            st.markdown("<br>", unsafe_allow_html=True)
            st.info(f"檔案內含 {len(sheet_names)} 張工作表")
            
        st.markdown("#### 🎯 請選擇您的分析模式：")
        analysis_mode = st.radio(
            "分析模式",
            [
                "模式一：一般數值預測 (單筆情境模擬，適用於無時間序列的資料)", 
                "模式二：時間序列趨勢預測 (未來區間走勢與基準對比，適用於營運/看診人數等時間序列)"
            ],
            label_visibility="collapsed"
        )
        
        raw_df = load_excel_data(uploaded_file, selected_sheet)
                
        st.session_state["df"] = raw_df
        st.session_state["selected_sheet"] = selected_sheet
        st.session_state["analysis_mode"] = analysis_mode

# ==================== 步驟二：欄位對應與模型訓練 ====================
if "df" in st.session_state:
    df = st.session_state["df"]
    analysis_mode = st.session_state["analysis_mode"]
    columns = df.columns.tolist()
    
    with st.expander("🔍 點擊展開：檢視目前載入的資料預覽 (前 5 筆)", expanded=False):
        st.dataframe(df.head(), width='stretch')
        
    st.markdown("---")
    st.markdown("### ⚙️ 步驟二：設定特徵並訓練模型")
    
    with st.container(border=True):
        if "模式一" in analysis_mode:
            st.info("💡 **目前為模式一**：已針對大數據優化訓練效能。")
            
            col_main, col_action = st.columns([2, 1])
            
            with col_main:
                numeric_cols = [col for col in columns if pd.api.types.is_numeric_dtype(df[col])]
                target_column = st.selectbox("請選擇預測目標欄位 (Target - 須為數值)", numeric_cols, key="m1_target")
                
                feature_candidates = [
                    col for col in columns 
                    if col != target_column and not pd.api.types.is_datetime64_any_dtype(df[col])
                ]
                selected_features = st.multiselect("請選擇影響預測的特徵變數", feature_candidates, default=feature_candidates, key="m1_feat")
            
            with col_action:
                st.markdown("<br><br>", unsafe_allow_html=True)
                train_btn = st.button("🚀 開始訓練模型\n(模式一)", type="primary", width='stretch')
            
            if train_btn:
                if not selected_features:
                    st.error("請至少選擇一個特徵變數！")
                else:
                    with st.spinner("⏳ 正在進行高效能資料清洗與 5 大模型軍團訓練，請稍候..."):
                        data_clean = df[selected_features + [target_column]].copy()
                        data_clean = data_clean.dropna(subset=[target_column])
                        
                        if len(data_clean) > 10000:
                            data_clean = data_clean.sample(n=10000, random_state=42)
                        
                        for col in selected_features:
                            if pd.api.types.is_numeric_dtype(data_clean[col]):
                                data_clean[col] = data_clean[col].fillna(data_clean[col].mean())
                            else:
                                data_clean[col] = data_clean[col].fillna("未知")
                                
                        X = data_clean[selected_features]
                        y = data_clean[target_column]
                        
                        X_encoded = pd.get_dummies(X, drop_first=True)
                        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
                        
                        models = {
                            "線性迴歸": LinearRegression(),
                            "隨機森林": RandomForestRegressor(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1),
                            "梯度提升樹": GradientBoostingRegressor(n_estimators=50, max_depth=6, random_state=42),
                            "XGBoost": XGBRegressor(n_estimators=50, max_depth=6, random_state=42, verbosity=0, n_jobs=-1),
                            "LightGBM": LGBMRegressor(n_estimators=50, max_depth=6, random_state=42, verbose=-1, n_jobs=-1)
                        }
                        
                        trained_models = {}
                        model_scores = {}
                        
                        for name, model in models.items():
                            model.fit(X_train, y_train)
                            trained_models[name] = model
                            y_pred_test = model.predict(X_test)
                            score = r2_score(y_test, y_pred_test)
                            model_scores[name] = round(score, 4)
                            
                        st.session_state["trained_models"] = trained_models
                        st.session_state["model_scores"] = model_scores
                        st.session_state["selected_features"] = selected_features
                        st.session_state["train_columns"] = X_encoded.columns.tolist()
                        st.session_state["df_original"] = df
                        st.session_state["mode"] = "mode1"
                        st.success("🎉 5 大模型軍團訓練完畢！")

        else:
            st.info("💡 **目前為模式二**：系統自動處理時間序列，產出未來區間帶狀走勢與基準對比。")
            
            detected_date_col = columns[0]
            for col in columns:
                if '日期' in str(col).lower() or 'date' in str(col).lower() or '時間' in str(col).lower() or 'yyy' in str(col).lower():
                    detected_date_col = col
                    break
            date_candidates = [detected_date_col] + [c for c in columns if c != detected_date_col]
            
            col_main, col_action = st.columns([2, 1])
            
            with col_main:
                date_column = st.selectbox("選擇日期/時間欄位 (Date)", date_candidates, key="m2_date")
                numeric_cols = [col for col in columns if col != date_column and pd.api.types.is_numeric_dtype(df[col])]
                target_column = st.selectbox("請選擇預測目標欄位 (Target - 須為數值)", numeric_cols, key="m2_target")
                
                feature_candidates = [col for col in columns if col != target_column and col != date_column]
                selected_features = st.multiselect("請選擇影響預測的特徵變數", feature_candidates, default=feature_candidates, key="m2_feat")
            
            with col_action:
                st.markdown("<br><br><br>", unsafe_allow_html=True)
                train_btn = st.button("🚀 開始訓練時間序列模型\n(模式二)", type="primary", width='stretch')
            
            if train_btn:
                if not selected_features:
                    st.error("請至少選擇一個特徵變數！")
                else:
                    with st.spinner("⏳ 正在解析時間序列與訓練模型，請稍候..."):
                        data_clean = df[selected_features + [target_column, date_column]].copy()
                        
                        sample_val = str(data_clean[date_column].dropna().iloc[0])
                        if len(sample_val) in [5, 6] and sample_val.isdigit():
                            def parse_roc_ym(val):
                                v_str = str(val).zfill(5)
                                roc_y = int(v_str[:-2])
                                m = int(v_str[-2:])
                                ad_y = roc_y + 1911
                                return pd.to_datetime(f"{ad_y}-{m:02d}-01", errors='coerce')
                            data_clean["_ParsedDate"] = data_clean[date_column].apply(parse_roc_ym)
                        else:
                            data_clean["_ParsedDate"] = pd.to_datetime(data_clean[date_column], errors='coerce')
                        
                        data_clean = data_clean.dropna(subset=[target_column, "_ParsedDate"])
                        
                        if len(data_clean) == 0:
                            st.error("❌ 錯誤：日期欄位解析失敗！請確認該欄位是否為正確的日期或 YYYMM 格式。")
                        else:
                            if len(data_clean) > 10000:
                                data_clean = data_clean.sample(n=10000, random_state=42)
                                
                            for col in selected_features:
                                if pd.api.types.is_numeric_dtype(data_clean[col]):
                                    data_clean[col] = data_clean[col].fillna(data_clean[col].mean())
                                else:
                                    data_clean[col] = data_clean[col].fillna("未知")
                                    
                            X = data_clean[selected_features]
                            y = data_clean[target_column]
                            
                            X_encoded = pd.get_dummies(X, drop_first=True)
                            X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
                            
                            models = {
                                "線性迴歸": LinearRegression(),
                                "隨機森林": RandomForestRegressor(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1),
                                "梯度提升樹": GradientBoostingRegressor(n_estimators=50, max_depth=6, random_state=42),
                                "XGBoost": XGBRegressor(n_estimators=50, max_depth=6, random_state=42, verbosity=0, n_jobs=-1),
                                "LightGBM": LGBMRegressor(n_estimators=50, max_depth=6, random_state=42, verbose=-1, n_jobs=-1)
                            }
                            
                            trained_models = {}
                            model_scores = {}
                            
                            for name, model in models.items():
                                model.fit(X_train, y_train)
                                trained_models[name] = model
                                y_pred_test = model.predict(X_test)
                                score = r2_score(y_test, y_pred_test)
                                model_scores[name] = round(score, 4)
                                
                            st.session_state["trained_models"] = trained_models
                            st.session_state["model_scores"] = model_scores
                            st.session_state["selected_features"] = selected_features
                            st.session_state["train_columns"] = X_encoded.columns.tolist()
                            st.session_state["df_original"] = df
                            st.session_state["date_column"] = date_column
                            st.session_state["target_column"] = target_column
                            st.session_state["mode"] = "mode2"
                            st.success("🎉 5 大時間序列模型軍團訓練完畢！")

# ==================== 步驟三：模型影響力與特徵重要程度解析 ====================
if "trained_models" in st.session_state:
    st.markdown("---")
    st.markdown("### 📊 關鍵影響因素分析 (變數重要程度排行榜)")
    
    with st.container(border=True):
        st.info("💡 **白話文解讀**：以下長條圖代表各個變數對預測結果的**影響力大小**。分數越高，代表這個項目對預測目標的影響越關鍵！")
        
        # 綜合隨機森林與 LightGBM 的特徵重要性來計算平均分數
        rf_model = st.session_state["trained_models"].get("隨機森林")
        lgb_model = st.session_state["trained_models"].get("LightGBM")
        
        feature_names = st.session_state["train_columns"]
        importance_dict = {feat: 0.0 for feat in feature_names}
        
        count = 0
        if rf_model is not None and hasattr(rf_model, "feature_importances_"):
            for f, imp in zip(feature_names, rf_model.feature_importances_):
                importance_dict[f] += imp
            count += 1
            
        if lgb_model is not None and hasattr(lgb_model, "feature_importances_"):
            for f, imp in zip(feature_names, lgb_model.feature_importances_):
                # LightGBM 重要性數值通常較大，這裡做個簡單歸一化加總
                total_lgb = sum(lgb_model.feature_importances_) if sum(lgb_model.feature_importances_) > 0 else 1
                importance_dict[f] += (imp / total_lgb)
            count += 1
            
        if count > 0:
            # 取平均並轉換成百分比或分數
            final_importance = {f: val / count for f, val in importance_dict.items()}
            imp_df = pd.DataFrame(list(final_importance.items()), columns=["變數名稱", "影響力分數"]).sort_values(by="影響力分數", ascending=True)
            
            # 限制最多顯示前 15 名重要的變數，畫面才不會過長
            if len(imp_df) > 15:
                imp_df = imp_df.tail(15)
                
            st.bar_chart(imp_df.set_index("變數名稱"))
        else:
            st.write("目前模型未支援特徵重要性分析。")

# ==================== 步驟四：情境模擬與預測結果檢視 ====================
if "trained_models" in st.session_state:
    st.markdown("---")
    st.markdown("### 🔮 步驟四：情境模擬與預測結果檢視")
    
    with st.container(border=True):
        current_mode = st.session_state.get("mode")
        features = st.session_state["selected_features"]
        df_orig = st.session_state["df_original"]
        
        if current_mode == "mode1":
            st.subheader("📊 單筆數值情境模擬 (支援未來年月自由輸入)")
            st.info("💡 **貼心設計**：若有 `YYYMM` 等欄位，系統已提供**自由文字輸入**功能！")
            
            with st.form("mode1_simulation_form"):
                override_features = st.multiselect(
                    "🔧 點此勾選您想要「手動覆寫/調整」的特徵",
                    features,
                    default=[]
                )
                
                input_overrides = {}
                if override_features:
                    st.markdown("#### 請設定您想調整的特徵數值/未來年月：")
                    for i in range(0, len(override_features), 2):
                        cols = st.columns(2)
                        for j in range(2):
                            if i + j < len(override_features):
                                feature = override_features[i + j]
                                with cols[j]:
                                    feat_series = df_orig[feature]
                                    if any(kw in feature.lower() for kw in ['yyy', 'ym', '年月', '代號', '年份', '月份']):
                                        latest_val = str(feat_series.dropna().iloc[-1]) if not feat_series.dropna().empty else "11507"
                                        input_overrides[feature] = st.text_input(f"輸入未來 {feature} (例如 11507)", value=latest_val, key=f"m1_txt_{feature}")
                                    elif pd.api.types.is_numeric_dtype(feat_series):
                                        default_val = float(feat_series.mean()) if not feat_series.dropna().empty else 0.0
                                        input_overrides[feature] = st.number_input(f"調整 {feature}", value=default_val, step=1.0, key=f"m1_ov_{feature}")
                                    else:
                                        unique_vals = feat_series.dropna().unique().tolist()
                                        if not unique_vals:
                                            unique_vals = ["預設"]
                                        input_overrides[feature] = st.selectbox(f"選擇 {feature}", unique_vals, key=f"m1_sel_{feature}")
                
                st.markdown("<br>", unsafe_allow_html=True)
                submitted = st.form_submit_button("💡 計算預測結果與模型對比", type="primary", width='stretch')
            
            if submitted:
                with st.spinner("⏳ 正在進行多模型情境推論中..."):
                    final_input_data = {}
                    for feature in features:
                        feat_series = df_orig[feature]
                        if pd.api.types.is_numeric_dtype(feat_series):
                            final_input_data[feature] = float(feat_series.mean())
                        else:
                            unique_vals = feat_series.dropna().unique().tolist()
                            final_input_data[feature] = unique_vals[0] if unique_vals else "預設"
                    
                    for feature, val in input_overrides.items():
                        final_input_data[feature] = val
                        
                    input_df = pd.DataFrame([final_input_data])
                    input_encoded = pd.get_dummies(input_df)
                    input_encoded = input_encoded.reindex(columns=st.session_state["train_columns"], fill_value=0)
                    
                    predictions = {name: model.predict(input_encoded)[0] for name, model in st.session_state["trained_models"].items()}
                    scores = st.session_state["model_scores"]
                    
                    pred_values = list(predictions.values())
                    min_val, max_val, avg_val = min(pred_values), max(pred_values), np.mean(pred_values)
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("📉 預測下限 (最低模型)", f"{min_val:.2f}")
                    c2.metric("📊 平均預測值 (綜合)", f"{avg_val:.2f}")
                    c3.metric("📈 預測上限 (最高模型)", f"{max_val:.2f}")
                    
                    st.markdown("#### 📋 各模型詳細預測數值與驗證準確率比較 (依 R² 排序)")
                    
                    detail_data = []
                    for name, pred in predictions.items():
                        detail_data.append({
                            "模型名稱": name,
                            "驗證準確率 (R² Score)": scores.get(name, 0.0),
                            "該模型預測數值": round(pred, 4)
                        })
                    
                    detail_df = pd.DataFrame(detail_data).sort_values(by="驗證準確率 (R² Score)", ascending=False).reset_index(drop=True)
                    st.dataframe(detail_df, width='stretch')

        elif current_mode == "mode2":
            st.subheader("📅 政策情境模擬：基準預測 vs. 方案調整對比")
            date_col = st.session_state["date_column"]
            target_col = st.session_state["target_column"]
            
            try:
                sample_val = str(df_orig[date_col].dropna().iloc[0])
                if len(sample_val) in [5, 6] and sample_val.isdigit():
                    def parse_roc_ym(val):
                        v_str = str(val).zfill(5)
                        return pd.to_datetime(f"{int(v_str[:-2]) + 1911}-{int(v_str[-2:]):02d}-01", errors='coerce')
                    df_orig["_TempDate"] = df_orig[date_col].apply(parse_roc_ym)
                else:
                    df_orig["_TempDate"] = pd.to_datetime(df_orig[date_col], errors='coerce')
                
                last_date = df_orig["_TempDate"].max().date()
                if pd.isna(last_date):
                    last_date = date.today()
            except:
                last_date = date.today()
                
            with st.form("mode2_simulation_form"):
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    start_date = st.date_input("選擇預測開始日期", value=last_date + timedelta(days=1))
                with col_d2:
                    end_date = st.date_input("選擇預測結束日期", value=last_date + timedelta(days=30))
                    
                st.markdown("#### ⚙️ 政策變數與類別選項調整")
                scenario_settings = {}
                
                for i in range(0, len(features), 3):
                    cols = st.columns(3)
                    for j in range(3):
                        if i + j < len(features):
                            feature = features[i + j]
                            with cols[j]:
                                if pd.api.types.is_numeric_dtype(df_orig[feature]):
                                    scenario_settings[feature] = st.slider(
                                        f"「{feature}」調整幅度 (%)", 
                                        min_value=-50, max_value=100, value=0, step=5, key=f"adj_{feature}"
                                    )
                                else:
                                    unique_vals = df_orig[feature].dropna().unique().tolist()
                                    if not unique_vals:
                                        unique_vals = ["預設"]
                                    scenario_settings[feature] = st.selectbox(
                                        f"選擇「{feature}」模擬類別", 
                                        unique_vals, 
                                        key=f"sel_cat_{feature}"
                                    )
                        
                st.markdown("<br>", unsafe_allow_html=True)
                submitted = st.form_submit_button("📊 產出基準對比與專業區間帶狀圖", type="primary", width='stretch')
                
            if start_date > end_date:
                st.error("錯誤：開始日期不能晚於結束日期！")
            elif submitted:
                with st.spinner("⏳ 正在計算未來區間動態模擬，請稍候..."):
                    df_orig["_IsWeekend"] = df_orig["_TempDate"].dt.weekday >= 5
                    
                    weekday_profile = df_orig[~df_orig["_IsWeekend"]][features].iloc[0].to_dict()
                    weekend_profile = df_orig[df_orig["_IsWeekend"]][features].iloc[0].to_dict()
                    
                    for feature in features:
                        if pd.api.types.is_numeric_dtype(df_orig[feature]):
                            weekday_profile[feature] = df_orig[~df_orig["_IsWeekend"]][feature].mean()
                            weekend_profile[feature] = df_orig[df_orig["_IsWeekend"]][feature].mean()
                        else:
                            unique_vals = df_orig[feature].dropna().unique().tolist()
                            weekday_profile[feature] = unique_vals[0] if unique_vals else "預設"
                            weekend_profile[feature] = unique_vals[0] if unique_vals else "預設"

                    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
                    trend_results = []
                    
                    for target_date in date_range:
                        is_weekend = target_date.weekday() >= 5
                        base_profile = weekend_profile if is_weekend else weekday_profile
                        
                        base_input = {}
                        scenario_input = {}
                        
                        for feature in features:
                            val = base_profile.get(feature, df_orig[feature].iloc[0])
                            
                            if pd.api.types.is_numeric_dtype(df_orig[feature]):
                                pct = scenario_settings.get(feature, 0)
                                base_input[feature] = val
                                scenario_input[feature] = val * (1 + pct / 100.0)
                            else:
                                base_input[feature] = val
                                scenario_input[feature] = scenario_settings.get(feature, val)
                            
                        base_df = pd.DataFrame([base_input])
                        base_encoded = pd.get_dummies(base_df).reindex(columns=st.session_state["train_columns"], fill_value=0)
                        base_preds = [model.predict(base_encoded)[0] for model in st.session_state["trained_models"].values()]
                        baseline_avg = np.mean(base_preds)
                        
                        scen_df = pd.DataFrame([scenario_input])
                        scen_encoded = pd.get_dummies(scen_df).reindex(columns=st.session_state["train_columns"], fill_value=0)
                        scen_preds = [model.predict(scen_encoded)[0] for model in st.session_state["trained_models"].values()]
                        
                        scen_min = min(scen_preds)
                        scen_avg = np.mean(scen_preds)
                        scen_max = max(scen_preds)
                        
                        trend_results.append({
                            "日期": target_date.strftime("%Y-%m-%d"),
                            "基準預測值 (維持現狀)": round(baseline_avg, 2),
                            "情境預測下限": round(scen_min, 2),
                            "情境平均預測值 (調整後)": round(scen_avg, 2),
                            "情境預測上限": round(scen_max, 2),
                            "效益增減量": round(scen_avg - baseline_avg, 2)
                        })
                        
                    trend_df = pd.DataFrame(trend_results).set_index("日期")
                    
                    st.markdown("### 📊 政策情境模擬與基準對比走勢圖")
                    try:
                        import plotly.graph_objects as go
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatter(
                            x=trend_df.index, y=trend_df["基準預測值 (維持現狀)"],
                            mode='lines', name='基準預測 (維持現狀)',
                            line=dict(color='orange', width=2, dash='dash')
                        ))
                        fig.add_trace(go.Scatter(
                            x=trend_df.index, y=trend_df["情境預測上限"],
                            mode='lines', name='情境預測上限 (方案最高)',
                            line=dict(color='rgba(0,180,255,0.5)', width=1),
                            legendgroup='range'
                        ))
                        fig.add_trace(go.Scatter(
                            x=trend_df.index, y=trend_df["情境預測下限"],
                            mode='lines', name='情境預測下限 (方案最低)',
                            line=dict(color='rgba(0,180,255,0.5)', width=1),
                            fill='tonexty',
                            fillcolor='rgba(0,180,255,0.15)',
                            legendgroup='range'
                        ))
                        fig.add_trace(go.Scatter(
                            x=trend_df.index, y=trend_df["情境平均預測值 (調整後)"],
                            mode='lines+markers', name='情境平均預測 (調整後中線)',
                            line=dict(color='rgb(0,120,255)', width=2.5)
                        ))
                        
                        fig.update_layout(
                            title="政策情境模擬 (方案調整 vs. 基準維持現狀)",
                            xaxis_title="日期",
                            yaxis_title="預測數值",
                            hovermode="x unified",
                            template="plotly_dark"
                        )
                        st.plotly_chart(fig, width='stretch')
                    except ImportError:
                        st.area_chart(trend_df[["基準預測值 (維持現狀)", "情境平均預測值 (調整後)"]])
                    
                    st.markdown("### 📋 未來各天詳細預測數據與總計對比")
                    
                    summary_row = pd.DataFrame({
                        "基準預測值 (維持現狀)": [trend_df["基準預測值 (維持現狀)"].sum()],
                        "情境預測下限": [trend_df["情境預測下限"].sum()],
                        "情境平均預測值 (調整後)": [trend_df["情境平均預測值 (調整後)"].sum()],
                        "情境預測上限": [trend_df["情境預測上限"].sum()],
                        "效益增減量": [trend_df["效益增減量"].sum()]
                    }, index=["總計 (Sum)"])
                    
                    display_df = pd.concat([trend_df, summary_row])
                    st.dataframe(display_df, width='stretch')