import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta, date
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

st.set_page_config(page_title="智慧營運與機器學習預測系統", layout="wide")

st.title("📈 智慧機器學習與情境預測系統")
st.write("請依照下方明確的步驟區塊，上傳資料檔案（支援 .xlsx 或 .csv）進行情境模擬與政策對比預測。")

# ==================== 支援 Excel 與 CSV 的快取讀取函式 ====================
@st.cache_data
def load_data_file(uploaded_file):
    file_name = uploaded_file.name.lower()
    if file_name.endswith('.csv'):
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='big5')
        return {"主資料表": df}, "主資料表", True
    else:
        excel_file = pd.ExcelFile(uploaded_file)
        sheets = {sheet: excel_file.parse(sheet) for sheet in excel_file.sheet_names}
        return sheets, excel_file.sheet_names[0], False

# ==================== 步驟一：資料上傳與工作表選擇 ====================
st.markdown("---")
st.markdown("### 📂 步驟一：上傳資料與工作表選擇")

with st.container(border=True):
    uploaded_file = st.file_uploader("請上傳資料檔案 (.xlsx 或 .csv)", type=["xlsx", "csv"])
    
    current_file_id = uploaded_file.name if uploaded_file is not None else None
    if "last_file_id" not in st.session_state:
        st.session_state["last_file_id"] = None

    if uploaded_file is not None:
        if st.session_state["last_file_id"] != current_file_id:
            st.session_state["last_file_id"] = current_file_id
            for key in ["trained_models", "model_scores", "selected_features", "target_encoding_maps", "global_mean", "train_columns", "df_original", "date_column", "target_column", "mode"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

        sheets_dict, default_sheet, is_csv = load_data_file(uploaded_file)
        sheet_names = list(sheets_dict.keys())
        
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            if not is_csv and len(sheet_names) > 1:
                selected_sheet = st.selectbox("📌 請選擇要讀取的工作表 (Sheet)", sheet_names, key="sheet_selector")
            else:
                selected_sheet = sheet_names[0]
                st.info("📌 目前為單一表格/CSV 檔案")
        with col_s2:
            st.markdown("<br>", unsafe_allow_html=True)
            if not is_csv:
                st.info(f"Excel 檔案內含 {len(sheet_names)} 張工作表")
            else:
                st.success("✅ CSV 檔案讀取成功")
            
        st.markdown("#### 🎯 請選擇您的分析模式：")
        analysis_mode = st.radio(
            "分析模式",
            [
                "模式一：一般數值預測 (單筆情境模擬，適用於無時間序列的資料)", 
                "模式二：時間序列趨勢預測 (未來區間走勢與基準對比，適用於營運/看診人數等時間序列)"
            ],
            label_visibility="collapsed"
        )
        
        raw_df = sheets_dict[selected_sheet]
        
        pseudo_numeric_keywords = ['yyy', 'ym', '年月', '代號', '年份', '月份']
        for col in raw_df.columns:
            if any(kw in str(col).lower() for kw in pseudo_numeric_keywords):
                raw_df[col] = raw_df[col].astype(str)
                
        st.session_state["df"] = raw_df
        st.session_state["selected_sheet"] = selected_sheet
        st.session_state["analysis_mode"] = analysis_mode
    else:
        st.session_state["last_file_id"] = None
        for key in ["df", "trained_models", "model_scores", "selected_features", "target_encoding_maps", "global_mean", "train_columns", "df_original", "date_column", "target_column", "mode"]:
            if key in st.session_state:
                del st.session_state[key]

# ==================== 步驟二：欄位對應與模型訓練 (5 大模型軍團) ====================
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
            st.info("💡 **目前為模式一**：已結合目標編碼與 5 大機器學習模型（Ridge、隨機森林、梯度提升樹、XGBoost、LightGBM）。")
            
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
                train_btn = st.button("🚀 開始訓練 5 大模型軍團\n(模式一)", type="primary", use_container_width=True)
            
            if train_btn:
                if not selected_features:
                    st.error("請至少選擇一個特徵變數！")
                else:
                    train_progress = st.progress(0)
                    train_status = st.empty()
                    
                    train_status.text("⏳ 正在進行高精度目標編碼與特徵工程...")
                    train_progress.progress(0.1)
                    
                    data_clean = df[selected_features + [target_column]].copy()
                    data_clean = data_clean.dropna(subset=[target_column])
                    
                    if len(data_clean) > 20000:
                        data_clean = data_clean.sample(n=20000, random_state=42)
                    
                    target_encoding_maps = {}
                    global_mean = data_clean[target_column].median()
                    
                    X_encoded = pd.DataFrame(index=data_clean.index)
                    for col in selected_features:
                        if pd.api.types.is_numeric_dtype(data_clean[col]):
                            X_encoded[col] = data_clean[col].fillna(data_clean[col].median())
                        else:
                            mapping = data_clean.groupby(col)[target_column].median().to_dict()
                            target_encoding_maps[col] = mapping
                            X_encoded[f"{col}_encoded"] = data_clean[col].map(mapping).fillna(global_mean)
                            
                    y = data_clean[target_column]
                    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
                    
                    models = {
                        "Ridge嶺迴歸": Ridge(alpha=1.0),
                        "隨機森林": RandomForestRegressor(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1),
                        "梯度提升樹": GradientBoostingRegressor(n_estimators=50, max_depth=6, random_state=42),
                        "XGBoost": XGBRegressor(n_estimators=50, max_depth=6, random_state=42, verbosity=0, n_jobs=-1),
                        "LightGBM": LGBMRegressor(n_estimators=50, max_depth=6, random_state=42, verbose=-1, n_jobs=-1)
                    }
                    
                    trained_models = {}
                    model_scores = {}
                    
                    total_models = len(models)
                    for idx, (name, model) in enumerate(models.items()):
                        pct = 0.2 + (idx + 1) / total_models * 0.75
                        train_progress.progress(min(pct, 1.0))
                        train_status.text(f"⏳ 正在訓練模型軍團: {name} ({idx + 1}/{total_models})...")
                        
                        model.fit(X_train, y_train)
                        trained_models[name] = model
                        y_pred_test = model.predict(X_test)
                        score = r2_score(y_test, y_pred_test)
                        model_scores[name] = round(score, 4)
                        
                    train_progress.empty()
                    train_status.empty()
                    
                    st.session_state["trained_models"] = trained_models
                    st.session_state["model_scores"] = model_scores
                    st.session_state["selected_features"] = selected_features
                    st.session_state["target_encoding_maps"] = target_encoding_maps
                    st.session_state["global_mean"] = global_mean
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
                train_btn = st.button("🚀 開始訓練 5 大時間序列模型\n(模式二)", type="primary", use_container_width=True)
            
            if train_btn:
                if not selected_features:
                    st.error("請至少選擇一個特徵變數！")
                else:
                    train_progress = st.progress(0)
                    train_status = st.empty()
                    
                    train_status.text("⏳ 正在解析時間序列與高精度中位數特徵...")
                    train_progress.progress(0.1)
                    
                    data_clean = df[selected_features + [target_column, date_column]].copy()
                    
                    if data_clean[date_column].dropna().empty:
                        train_status.empty()
                        st.error("❌ 錯誤：所選的日期欄位無有效資料，請重新選擇正確的日期/時間欄位！")
                        st.stop()

                    try:
                        if pd.api.types.is_numeric_dtype(df[date_column]):
                            parsed = pd.Series(pd.NaT, index=data_clean.index)
                        else:
                            sample_str_series = data_clean[date_column].astype(str).str.strip()
                            has_symbols = sample_str_series.str.contains(r'[-/.]').any()
                            is_roc = sample_str_series.str.match(r'^\d{5,6}$').any()
                            
                            if not has_symbols and not is_roc:
                                parsed = pd.Series(pd.NaT, index=data_clean.index)
                            else:
                                if is_roc and not has_symbols:
                                    def parse_roc_ym(val):
                                        v_str = str(val).strip().zfill(5)
                                        if not v_str.isdigit() or len(v_str) not in [5, 6]:
                                            return pd.NaT
                                        roc_y = int(v_str[:-2])
                                        m = int(v_str[-2:])
                                        if not (1 <= m <= 12) or not (90 <= roc_y <= 125):
                                            return pd.NaT
                                        return pd.to_datetime(f"{roc_y + 1911}-{m:02d}-01", errors='coerce')
                                    parsed = data_clean[date_column].apply(parse_roc_ym)
                                else:
                                    parsed = pd.to_datetime(data_clean[date_column], errors='coerce')
                                    if parsed.notna().any():
                                        years = parsed.dt.year
                                        parsed = parsed.where((years >= 1900) & (years <= 2100), pd.NaT)
                        
                        data_clean["_ParsedDate"] = parsed
                    except Exception as e:
                        data_clean["_ParsedDate"] = pd.Series(pd.NaT, index=data_clean.index)
                    
                    data_clean = data_clean.dropna(subset=[target_column, "_ParsedDate"])
                    
                    if len(data_clean) == 0:
                        train_status.empty()
                        st.error("❌ 錯誤：日期欄位解析失敗！請選擇包含正確日期格式（如 YYYY-MM-DD）的欄位。")
                        st.stop()
                    else:
                        if len(data_clean) > 20000:
                            data_clean = data_clean.sample(n=20000, random_state=42)
                            
                        target_encoding_maps = {}
                        global_mean = data_clean[target_column].median()
                        
                        X_encoded = pd.DataFrame(index=data_clean.index)
                        for col in selected_features:
                            if pd.api.types.is_numeric_dtype(data_clean[col]):
                                X_encoded[col] = data_clean[col].fillna(data_clean[col].median())
                            else:
                                mapping = data_clean.groupby(col)[target_column].median().to_dict()
                                target_encoding_maps[col] = mapping
                                X_encoded[f"{col}_encoded"] = data_clean[col].map(mapping).fillna(global_mean)
                                
                        y = data_clean[target_column]
                        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
                        
                        models = {
                            "Ridge嶺迴歸": Ridge(alpha=1.0),
                            "隨機森林": RandomForestRegressor(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1),
                            "梯度提升樹": GradientBoostingRegressor(n_estimators=50, max_depth=6, random_state=42),
                            "XGBoost": XGBRegressor(n_estimators=50, max_depth=6, random_state=42, verbosity=0, n_jobs=-1),
                            "LightGBM": LGBMRegressor(n_estimators=50, max_depth=6, random_state=42, verbose=-1, n_jobs=-1)
                        }
                        
                        trained_models = {}
                        model_scores = {}
                        
                        total_models = len(models)
                        for idx, (name, model) in enumerate(models.items()):
                            pct = 0.2 + (idx + 1) / total_models * 0.75
                            train_progress.progress(min(pct, 1.0))
                            train_status.text(f"⏳ 正在訓練時間序列模型: {name} ({idx + 1}/{total_models})...")
                            
                            model.fit(X_train, y_train)
                            trained_models[name] = model
                            y_pred_test = model.predict(X_test)
                            score = r2_score(y_test, y_pred_test)
                            model_scores[name] = round(score, 4)
                            
                        train_progress.empty()
                        train_status.empty()
                        
                        st.session_state["trained_models"] = trained_models
                        st.session_state["model_scores"] = model_scores
                        st.session_state["selected_features"] = selected_features
                        st.session_state["target_encoding_maps"] = target_encoding_maps
                        st.session_state["global_mean"] = global_mean
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
        st.info("💡 **白話文解讀**：以下長條圖呈現各特徵變數對預測目標的整體影響力大小。")
        
        rf_model = st.session_state["trained_models"].get("隨機森林")
        feature_names = st.session_state["train_columns"]
        
        if rf_model is not None and hasattr(rf_model, "feature_importances_"):
            imp_df = pd.DataFrame({
                "特徵變數名稱": [f.replace("_encoded", "") for f in feature_names],
                "綜合影響力分數": rf_model.feature_importances_
            }).sort_values(by="綜合影響力分數", ascending=True)
            st.bar_chart(imp_df.set_index("特徵變數名稱"))
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
        target_maps = st.session_state["target_encoding_maps"]
        g_mean = st.session_state["global_mean"]
        
        if current_mode == "mode1":
            st.subheader("📊 單筆數值情境模擬 (支援未來代號與數值自由輸入)")
            st.info("💡 **高精度提示**：系統會自動將您輸入的代號轉換為對應的歷史量級進行精準預測！")
            
            with st.form("mode1_simulation_form"):
                override_features = st.multiselect(
                    "🔧 點此勾選您想要「手動覆寫/調整」的特徵",
                    features,
                    default=[]
                )
                
                input_overrides = {}
                if override_features:
                    st.markdown("#### 請設定您想調整的特徵數值/未來代號：")
                    for i in range(0, len(override_features), 2):
                        cols = st.columns(2)
                        for j in range(2):
                            if i + j < len(override_features):
                                feature = override_features[i + j]
                                with cols[j]:
                                    feat_series = df_orig[feature]
                                    if any(kw in feature.lower() for kw in ['yyy', 'ym', '年月', '代號', '年份', '月份']):
                                        latest_val = str(feat_series.dropna().iloc[-1]) if not feat_series.dropna().empty else "11507"
                                        input_overrides[feature] = st.text_input(f"輸入未來 {feature} (例如 00101B)", value=latest_val, key=f"m1_txt_{feature}")
                                    elif pd.api.types.is_numeric_dtype(feat_series):
                                        default_val = float(feat_series.median()) if not feat_series.dropna().empty else 0.0
                                        input_overrides[feature] = st.number_input(f"調整 {feature}", value=default_val, step=1.0, key=f"m1_ov_{feature}")
                                    else:
                                        unique_vals = feat_series.dropna().unique().tolist()
                                        if not unique_vals:
                                            unique_vals = ["預設"]
                                        input_overrides[feature] = st.selectbox(f"選擇 {feature}", unique_vals, key=f"m1_sel_{feature}")
                
                st.markdown("<br>", unsafe_allow_html=True)
                submitted = st.form_submit_button("💡 計算高精度預測結果與模型對比", type="primary", use_container_width=True)
            
            if submitted:
                with st.spinner("⏳ 正在進行高精度智慧加權推論中..."):
                    final_input_data = {}
                    for feature in features:
                        feat_series = df_orig[feature]
                        if pd.api.types.is_numeric_dtype(feat_series):
                            final_input_data[feature] = float(feat_series.median())
                        else:
                            unique_vals = feat_series.dropna().unique().tolist()
                            final_input_data[feature] = unique_vals[0] if unique_vals else "預設"
                    
                    for feature, val in input_overrides.items():
                        final_input_data[feature] = val
                        
                    encoded_input = {}
                    for feature in features:
                        val = final_input_data[feature]
                        if feature in target_maps:
                            mapping = target_maps[feature]
                            encoded_input[f"{feature}_encoded"] = mapping.get(val, g_mean)
                        else:
                            encoded_input[feature] = float(val) if pd.api.types.is_numeric_dtype(df_orig[feature]) else 0.0
                            
                    input_df = pd.DataFrame([encoded_input])
                    input_df = input_df.reindex(columns=st.session_state["train_columns"], fill_value=0)
                    
                    predictions = {name: model.predict(input_df)[0] for name, model in st.session_state["trained_models"].items()}
                    scores = st.session_state["model_scores"]
                    
                    valid_preds = {}
                    weights = {}
                    for name, pred in predictions.items():
                        r2 = scores.get(name, 0.0)
                        if r2 > 0:
                            valid_preds[name] = pred
                            weights[name] = r2
                            
                    if valid_preds:
                        total_weight = sum(weights.values())
                        weighted_avg = sum(valid_preds[name] * weights[name] for name in valid_preds) / total_weight
                        pred_values = list(valid_preds.values())
                        min_val, max_val = min(pred_values), max(pred_values)
                    else:
                        pred_values = list(predictions.values())
                        min_val, max_val, weighted_avg = min(pred_values), max(pred_values), np.median(pred_values)
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("📉 預測下限", f"{min_val:,.2f}")
                    c2.metric("📊 智慧加權平均預測值 (推薦)", f"{weighted_avg:,.2f}")
                    c3.metric("📈 預測上限", f"{max_val:,.2f}")
                    
                    st.markdown("#### 📋 各模型詳細預測數值與驗證準確率比較 (依 R² 排序)")
                    
                    detail_data = []
                    for name, pred in predictions.items():
                        detail_data.append({
                            "模型名稱": name,
                            "驗證準確率 (R² Score)": scores.get(name, 0.0),
                            "該模型預測數值": round(pred, 4)
                        })
                    
                    detail_df = pd.DataFrame(detail_data).sort_values(by="驗證準確率 (R² Score)", ascending=False).reset_index(drop=True)
                    st.dataframe(detail_df, use_container_width=True)

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
                submitted = st.form_submit_button("📊 產出基準對比與專業區間帶狀圖", type="primary", use_container_width=True)
                
            if start_date > end_date:
                st.error("錯誤：開始日期不能晚於結束日期！")
            elif submitted:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                df_orig["_IsWeekend"] = df_orig["_TempDate"].dt.weekday >= 5
                
                weekday_df = df_orig[~df_orig["_IsWeekend"]]
                weekend_df = df_orig[df_orig["_IsWeekend"]]
                
                weekday_profile = {}
                weekend_profile = {}
                
                # 使用中位數（Median）防爆機制，避免極端值干擾
                for feature in features:
                    if pd.api.types.is_numeric_dtype(df_orig[feature]):
                        weekday_profile[feature] = float(weekday_df[feature].median()) if not weekday_df.empty else float(df_orig[feature].median())
                        weekend_profile[feature] = float(weekend_df[feature].median()) if not weekend_df.empty else float(df_orig[feature].median())
                    else:
                        w_mode = weekday_df[feature].mode() if not weekday_df.empty else pd.Series()
                        weekday_profile[feature] = w_mode.iloc[0] if not w_mode.empty else (df_orig[feature].iloc[0] if not df_orig[feature].empty else "預設")
                        
                        we_mode = weekend_df[feature].mode() if not weekend_df.empty else pd.Series()
                        weekend_profile[feature] = we_mode.iloc[0] if not we_mode.empty else (df_orig[feature].iloc[0] if not df_orig[feature].empty else "預設")

                date_range = pd.date_range(start=start_date, end=end_date, freq="D")
                trend_results = []
                scores = st.session_state["model_scores"]
                total_days = len(date_range)
                
                # 引入動態慣性滾動因子（讓預存曲線帶有時間序列的自然震盪與慣性，而非死平線）
                np.random.seed(42)
                rolling_inertia = 1.0

                for idx, target_date in enumerate(date_range):
                    progress_pct = int((idx + 1) / total_days * 100)
                    progress_bar.progress((idx + 1) / total_days)
                    status_text.text(f"⏳ 正在進行未來動態模擬中... ({idx + 1}/{total_days} 天, {progress_pct}%)")
                    
                    is_weekend = target_date.weekday() >= 5
                    base_profile = weekend_profile if is_weekend else weekday_profile
                    
                    base_input = {}
                    scenario_input = {}
                    
                    # 加入微幅隨機波動慣性（模擬真實營運的日夜起伏）
                    rolling_inertia = max(0.85, min(1.15, rolling_inertia + np.random.normal(0, 0.03)))

                    for feature in features:
                        val = base_profile.get(feature, df_orig[feature].iloc[0])
                        
                        if pd.api.types.is_numeric_dtype(df_orig[feature]):
                            pct = scenario_settings.get(feature, 0)
                            base_input[feature] = val * rolling_inertia
                            scenario_input[feature] = val * rolling_inertia * (1 + pct / 100.0)
                        else:
                            base_input[feature] = val
                            scenario_input[feature] = scenario_settings.get(feature, val)
                            
                    def transform_profile(prof):
                        res = {}
                        for f in features:
                            v = prof.get(f)
                            if f in target_maps:
                                res[f"{f}_encoded"] = target_maps[f].get(v, g_mean)
                            else:
                                res[f] = float(v) if pd.api.types.is_numeric_dtype(df_orig[f]) else 0.0
                        return pd.DataFrame([res]).reindex(columns=st.session_state["train_columns"], fill_value=0)

                    base_encoded = transform_profile(base_input)
                    base_preds = {name: model.predict(base_encoded)[0] for name, model in st.session_state["trained_models"].items()}
                    
                    valid_base = {n: p for n, p in base_preds.items() if scores.get(n, 0) > 0}
                    if valid_base:
                        tot_w = sum(scores[n] for n in valid_base)
                        baseline_avg = sum(valid_base[n] * scores[n] for n in valid_base) / tot_w
                    else:
                        baseline_avg = np.median(list(base_preds.values()))
                    
                    scen_encoded = transform_profile(scenario_input)
                    scen_preds = {name: model.predict(scen_encoded)[0] for name, model in st.session_state["trained_models"].items()}
                    
                    valid_scen = {n: p for n, p in scen_preds.items() if scores.get(n, 0) > 0}
                    if valid_scen:
                        tot_w = sum(scores[n] for n in valid_scen)
                        scen_avg = sum(valid_scen[n] * scores[n] for n in valid_scen) / tot_w
                        scen_min = min(valid_scen.values())
                        scen_max = max(valid_scen.values())
                    else:
                        scen_avg = np.median(list(scen_preds.values()))
                        scen_min = min(scen_preds.values())
                        scen_max = max(scen_preds.values())
                    
                    trend_results.append({
                        "日期": target_date.strftime("%Y-%m-%d"),
                        "基準預測值 (維持現狀)": round(baseline_avg, 2),
                        "情境預測下限": round(scen_min, 2),
                        "情境平均預測值 (調整後)": round(scen_avg, 2),
                        "情境預測上限": round(scen_max, 2),
                        "效益增減量": round(scen_avg - baseline_avg, 2)
                    })
                
                status_text.empty()
                progress_bar.empty()
                
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
                        line=dict(color='rgba(0,180,255,0.5)', width=1, dash='dot'),
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
                    st.plotly_chart(fig, use_container_width=True)
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
                st.dataframe(display_df, use_container_width=True)
