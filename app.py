import streamlit as st
import pandas as pd
import numpy as np
from fuzzywuzzy import process
from typing import List, Tuple, Optional
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, f1_score, accuracy_score, recall_score, precision_recall_curve, precision_score, classification_report
import plotly.graph_objects as go
import plotly.figure_factory as ff

def load_data(file) -> Optional[pd.DataFrame]:
    try:
        if file.name.endswith('.csv'):
            return pd.read_csv(file)
        elif file.name.endswith(('.xls', '.xlsx')):
            return pd.read_excel(file)
        else:
            st.error("지원하지 않는 파일 형식입니다. CSV 또는 Excel 파일을 업로드해주세요.")
            return None
    except Exception as e:
        st.error(f"파일 로드 중 오류 발생: {str(e)}")
        return None

def find_column(keywords: List[str], df: pd.DataFrame, threshold: int = 60) -> Optional[str]:
    best_match = None
    max_score = 0
    for keyword in keywords:
        matches = process.extractOne(keyword, df.columns)
        if matches and matches[1] > max_score:
            best_match = matches[0]
            max_score = matches[1]
    return best_match if max_score >= threshold else None

def is_binary(series: pd.Series) -> bool:
    return series.nunique() == 2

def is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)

def recommend_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    ground_truth_keywords = ['뇌출혈여부', '정답', '실제값', 'ground_truth', 'label', 'target']
    prediction_keywords = ['AI스코어', '예측', 'prediction', 'score', 'pred', 'output']
    
    pred_col = find_column(prediction_keywords, df)
    gt_col = find_column(ground_truth_keywords, df)
    
    if pred_col is None:
        pred_candidates = [col for col in df.columns if is_numeric(df[col]) and 
                           ((0 <= df[col].min() <= df[col].max() <= 1) or 
                            (0 <= df[col].min() <= df[col].max() <= 100))]
        pred_col = pred_candidates[0] if pred_candidates else None
    
    if gt_col is None:
        gt_candidates = [col for col in df.columns if is_binary(df[col])]
        gt_col = gt_candidates[0] if gt_candidates else None
    
    return pred_col, gt_col

def validate_pred_column(df: pd.DataFrame, pred_col: str) -> Tuple[bool, str, float, float]:
    if pred_col is None:
        return False, "예측 컬럼을 선택해주세요.", 0, 1
    
    if not is_numeric(df[pred_col]):
        return False, f"예측 컬럼 '{pred_col}'이 수치형이 아닙니다.", 0, 1
    
    pred_min, pred_max = df[pred_col].min(), df[pred_col].max()
    
    if 0 <= pred_min <= pred_max <= 1:
        scale = "0-1"
        default_threshold = 0.5
    elif 0 <= pred_min <= pred_max <= 100:
        scale = "0-100"
        default_threshold = 50
    else:
        return False, f"예측 컬럼 '{pred_col}'의 값이 0~1 또는 0~100 범위를 벗어납니다. (현재 범위: {pred_min:.2f}~{pred_max:.2f})", pred_min, pred_max
    
    return True, f"예측 컬럼 '{pred_col}'은 {scale} 스케일입니다.", pred_min, pred_max

def validate_gt_column(df: pd.DataFrame, gt_col: str) -> Tuple[bool, str]:
    if gt_col is None:
        return False, "정답 컬럼을 선택해주세요."
    
    if not is_binary(df[gt_col]):
        return False, f"정답 컬럼 '{gt_col}'이/가 이진 분류가 아닙니다."
    
    return True, f"정답 컬럼 '{gt_col}'이/가 이진 분류입니다."

def display_data_info(df: pd.DataFrame):
    st.subheader("데이터셋 정보")
    st.write(f"총 행 수: {df.shape[0]}, 총 열 수: {df.shape[1]}")
    if st.checkbox("열 정보 표시"):
        col_info = []
        for col in df.columns:
            col_type = df[col].dtype
            unique_values = df[col].nunique()
            sample_values = df[col].sample(min(5, len(df))).tolist()
            
            if pd.api.types.is_numeric_dtype(df[col]):
                try:
                    value_range = f"{df[col].min()} ~ {df[col].max()}"
                except TypeError:
                    value_range = "N/A (Mixed numeric types)"
            else:
                value_range = "N/A"
            
            col_info.append({
                "컬럼명": col,
                "데이터 타입": col_type,
                "고유값 수": unique_values,
                "값 범위": value_range
                #,"샘플 값": str(sample_values)[:100]  # 샘플 값을 문자열로 변환하고 길이 제한
            })
        
        st.table(pd.DataFrame(col_info))

def analyze_performance(df: pd.DataFrame, pred_col: str, gt_col: str, positive_label, threshold: float):
    try:
        y_true = (df[gt_col] == positive_label).astype(int)
        y_score = df[pred_col].astype(float)

        # 스코어 범위 확인 및 정규화
        score_min, score_max = y_score.min(), y_score.max()
        if score_max > 1:
            y_score = y_score / 100  # 0-100 스케일을 0-1로 변환
            threshold = threshold / 100  # 임계값도 함께 변환

        # ROC AUC
        roc_auc = roc_auc_score(y_true, y_score)
        st.write(f"ROC AUC: {roc_auc:.4f}")
        
        # 스코어를 이진 예측으로 변환
        y_pred = (y_score > threshold).astype(int)

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Confusion Matrix 시각화
        labels = ['뇌출혈 음성', '뇌출혈 양성']
        cm_fig = ff.create_annotated_heatmap(
            z=cm, x=labels, y=labels, colorscale='Blues',
            annotation_text=cm
        )
        cm_fig.update_layout(title_text='Confusion Matrix', xaxis_title='예측', yaxis_title='실제')
        st.plotly_chart(cm_fig)

        # Classification Report
        class_report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
        df_class_report = pd.DataFrame(class_report).transpose()
        st.markdown("**Classification Report:**")
        st.dataframe(df_class_report.style.format("{:.4f}"))

        # Performance Metrics
        metrics = {
            "민감도 (Sensitivity/Recall)": recall_score(y_true, y_pred),
            "특이도 (Specificity)": cm[0, 0] / (cm[0, 0] + cm[0, 1]),
            "정확도 (Accuracy)": accuracy_score(y_true, y_pred),
            "정밀도 (Precision)": precision_score(y_true, y_pred),
            "F1 Score": f1_score(y_true, y_pred)
        }

        for metric, value in metrics.items():
            st.write(f"{metric}: {value:.4f}")

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_score)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (AUC = {roc_auc:.4f})'))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
        fig.add_trace(go.Scatter(x=[fpr[np.argmin(np.abs(tpr - threshold))]], 
                                 y=[tpr[np.argmin(np.abs(tpr - threshold))]], 
                                 mode='markers', 
                                 name=f'Threshold ({threshold:.2f})', 
                                 marker=dict(size=10)))
        fig.update_layout(
            title='Receiver Operating Characteristic (ROC) Curve',
            xaxis_title='위양성률 (False Positive Rate)',
            yaxis_title='민감도 (True Positive Rate)',
            legend=dict(x=0.1, y=0.9),
            width=700,
            height=500
        )
        st.plotly_chart(fig)

        # 예측 분포 시각화
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=y_score[y_true == 0], name='뇌출혈 음성', opacity=0.7))
        fig.add_trace(go.Histogram(x=y_score[y_true == 1], name='뇌출혈 양성', opacity=0.7))
        fig.add_trace(go.Scatter(x=[threshold, threshold], y=[0, y_score.shape[0]//2], 
                                 mode='lines', name='임계값', 
                                 line=dict(color='red', width=2, dash='dash')))
        fig.update_layout(
            title='예측 스코어 분포',
            xaxis_title='예측 스코어',
            yaxis_title='빈도',
            barmode='overlay',
            width=700,
            height=400
        )
        st.plotly_chart(fig)

    except Exception as e:
        st.error(f"분석 중 오류가 발생했습니다: {str(e)}")
        st.write("오류 상세 정보:")
        st.write(e)
        st.write("데이터 및 선택한 컬럼을 다시 확인해주세요.")

def main():
    st.title('분류모델 성능 시각화')
    
    # 세션 상태 초기화
    if 'file_uploaded' not in st.session_state:
        st.session_state.file_uploaded = False
    
    uploaded_file = st.file_uploader("CSV 또는 Excel 파일 업로드", type=['csv', 'xlsx', 'xls'])
    
    if uploaded_file is not None:
        # 파일이 새로 업로드되면 세션 상태를 리셋
        if st.session_state.file_uploaded and uploaded_file.name != st.session_state.last_uploaded_file:
            st.session_state.clear()
            st.experimental_rerun()
        
        st.session_state.file_uploaded = True
        st.session_state.last_uploaded_file = uploaded_file.name
        
        df = load_data(uploaded_file)
        if df is not None:
            st.success("파일이 성공적으로 업로드되었습니다!")
            
            st.subheader("데이터 샘플 (처음 10개 행)")
            st.dataframe(df.head(10))
            
            display_data_info(df)
            
            pred_col, gt_col = recommend_columns(df)
            
            st.subheader("컬럼 선택 및 검증")
            all_columns = df.columns.tolist()
            
            col1, col2 = st.columns(2)
            
            with col1:
                selected_pred_col = st.selectbox("AI Score 컬럼", all_columns, index=all_columns.index(pred_col) if pred_col in all_columns else 0, key='pred_col')
                pred_valid, pred_message, pred_min, pred_max = validate_pred_column(df, selected_pred_col)
                if pred_valid:
                    st.success(pred_message)
                else:
                    st.error(pred_message)
            
            with col2:
                selected_gt_col = st.selectbox("Ground Truth 컬럼", all_columns, index=all_columns.index(gt_col) if gt_col in all_columns else 0, key='gt_col')
                gt_valid, gt_message = validate_gt_column(df, selected_gt_col)
                if gt_valid:
                    st.success(gt_message)
                else:
                    st.error(gt_message)
            
            if pred_valid and gt_valid:
                st.subheader("분석 설정")
                
                # 양성 레이블 선택
                unique_values = df[selected_gt_col].unique()
                positive_label = st.selectbox("양성(뇌출혈) 레이블 선택", unique_values, index=0 if len(unique_values) > 1 else 0)
                
                # 임계값 설정
                if pred_max <= 1:
                    default_threshold = 0.5
                    threshold = st.slider("예측 임계값", 0.0, 1.0, float(default_threshold), 0.01)
                else:
                    default_threshold = 50.0
                    threshold = st.slider("예측 임계값", 0.0, 100.0, float(default_threshold), 1.0)                
                if st.button("분석 실행"):
                    st.subheader("성능 분석 결과")
                    analyze_performance(df, selected_pred_col, selected_gt_col, positive_label, threshold)
            else:
                st.error("유효한 AI Score 컬럼과 Ground Truth 컬럼을 선택해주세요.")


if __name__ == "__main__":
    main()
    
    
"""
--------------------------------------------------------------------
파일 업로드
 - AI 분석 결과와 정답이 포함된 csv, excel 등 파일을 사용자가 선택 -> 로드
 - 업로드한 파일명과 성공 여부 확인
 - 데이터를 파악할 수 있게 보여주기 (기능 숨김 디폴트)
    - 10개 row 표시
    - 컬럼 수와 컬럼 속성 표시
    - (미구현)데이터 차트 미리보기-컬럼 속성에 따라 적절한 plot 선택

분석을 위한 사전 설정
 - 분석 대상 컬럼 선택
    - 자동으로 AI score와 정답 컬럼 추천
    - 사용자가 드롭다운에서 선택 가능하며 분석 시 확정 (컬럼 load)
 - 정답 컬럼의 양성 레이블 선택 (정답 컬럼 load)
 - 양성 판정 임계값 설정

분석
 - 위 설정한 정보로 분석 시도
 - 분석 실패 시 재시도
 - Confusion Matrix
 - Classification Report
 - 예측 스코어 분포
 
 - (미구현)사전에 분석에 사용할 데이터 수를 순차적으로 또는 랜덤으로 선택하도록 함. (원하는 성능을 내도록 조정이 필요한 경우)
데이터 시각화
 
 - 컬럼 속성에 따른 디스플레이 방법은 chatGPT 의존하거나, 룰베이스로 자동 시각화하도록 구성. -> chatGPT 의존도 낮춰야할 것

(미구현)자동 레이블링-필요시
 - 이제 이것도 한 번에 던져서 결과를 json으로 받을 수 있겠다. 한 건씩 하는게 아니라.
"""