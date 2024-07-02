import streamlit as st
import pandas as pd
import numpy as np
from fuzzywuzzy import process
from typing import List, Tuple, Optional
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, f1_score, accuracy_score, recall_score, precision_recall_curve, precision_score, classification_report, cohen_kappa_score
import plotly.graph_objects as go
import plotly.figure_factory as ff
from scipy import stats
from scipy.stats import chi2
from scipy.stats import fisher_exact, chi2_contingency

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
    
    return True, f"예측 컬럼 '{pred_col}'is {scale} 스케일입니다.", pred_min, pred_max

def validate_gt_column(df: pd.DataFrame, gt_col: str) -> Tuple[bool, str]:
    if gt_col is None:
        return False, "정답 컬럼을 선택해주세요."
    
    if not is_binary(df[gt_col]):
        return False, f"정답 컬럼 '{gt_col}'is 이진 분류가 아닙니다."
    
    return True, f"정답 컬럼 '{gt_col}'is 이진 분류입니다."

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

def accuracy_binomial_test(y_true, y_pred):
    n = len(y_true)
    correct = sum(y_true == y_pred)
    accuracy = correct / n
    p_value = 2 * min(stats.binom.cdf(correct, n, 0.5), 1 - stats.binom.cdf(correct - 1, n, 0.5))
    return {
        "method": "정확도에 대한 이항 검정",
        "description": "모델의 전체 정확도가 우연에 의한 것인지 평가합니다.",
        "accuracy": accuracy,
        "p_value": p_value,
        "significance": "statistically significant" if p_value < 0.05 else "not statistically significant"
    }

def auc_confidence_interval(y_true, y_score, alpha=0.05):
    auc = roc_auc_score(y_true, y_score)
    n1 = sum(y_true)
    n2 = len(y_true) - n1
    q1 = auc / (2 - auc)
    q2 = 2 * auc ** 2 / (1 + auc)
    se = np.sqrt((auc * (1 - auc) + (n1 - 1) * (q1 - auc ** 2) + (n2 - 1) * (q2 - auc ** 2)) / (n1 * n2))
    ci_lower = auc - stats.norm.ppf(1 - alpha / 2) * se
    ci_upper = auc + stats.norm.ppf(1 - alpha / 2) * se
    return {
        "method": "AUC에 대한 신뢰구간 계산",
        "description": "ROC 곡선의 AUC에 대한 신뢰성을 평가합니다.",
        "auc": auc,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper
    }

def confusion_matrix_chi2_test(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    chi2, p_value, dof, expected = chi2_contingency(cm)
    return {
        "method": "혼동 행렬에 대한 카이-제곱 검정",
        "description": "예측 결과와 실제 값 사이의 관계가 통계적으로 유의미한지 평가합니다.",
        "chi2": chi2,
        "p_value": p_value,
        "significance": "statistically significant" if p_value < 0.05 else "not statistically significant"
    }

def mcnemar_test(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    statistic = (abs(fp - fn) - 1)**2 / (fp + fn)
    p_value = chi2.sf(statistic, 1)

    return {
        "method": "McNemar's Test",
        "description": "모델의 예측과 실제 값 사이의 차이가 통계적으로 유의미한지 평가합니다. 여기서는 단일 모델의 성능을 평가하는 데 사용됩니다.",
        "statistic": statistic,
        "p_value": p_value,
        "significance": "statistically significant" if p_value < 0.05 else "not statistically significant"
    }

def cohen_kappa(y_true, y_pred):
    kappa = cohen_kappa_score(y_true, y_pred)
    return {
        "method": "Cohen's Kappa",
        "description": "모델의 예측과 실제 값 사이의 일치도를 평가합니다. 일반적으로 두 평가자 간의 일치도를 측정하지만, 여기서는 모델 예측과 실제 값의 일치도를 측정합니다. (일치도가 1이면 완전 일치)",
        "kappa": kappa
    }

def fishers_exact_test(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    table = np.array([[tn, fp], [fn, tp]])
    odds_ratio, p_value = fisher_exact(table)
    return {
        "method": "Fisher's Exact Test",
        "description": "소규모 데이터셋에서 혼동 행렬을 기반으로 민감도, 특이도 등의 성능 지표에 대한 p값을 계산합니다.",
        "p_value": p_value,
        "significance": "statistically significant" if p_value < 0.05 else "not statistically significant"
    }

def analyze_performance(df: pd.DataFrame, pred_col: str, gt_col: str, positive_label, threshold: float):
    try:
        y_true = (df[gt_col] == positive_label).astype(int)
        y_score = df[pred_col].astype(float)
        
        # 스코어 범위 확인 및 정규화
        # score_min, score_max = y_score.min(), y_score.max()
        # if score_max > 1:
        #     y_score = y_score / 100  # 0-100 스케일을 0-1로 변환
        #     threshold = threshold / 100  # 임계값도 함께 변환
        
        # 스코어를 threshold 기준으로 이진 예측으로 변환 (임계값 이상인 경우 양성)
        y_pred = (y_score >= threshold).astype(int)

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Confusion Matrix 시각화
        
        labels = ['음성', '양성']
        label_size=20
        cm_fig = ff.create_annotated_heatmap(
            z=cm, x=labels, y=labels, colorscale='Blues',
            annotation_text=cm
        )
        # cm_fig.update_layout(title_text='Confusion Matrix', xaxis_title='예측', yaxis_title='실제')
        cm_fig.update_layout(
            title_text='Confusion Matrix', 
            xaxis_title='예측', 
            yaxis_title='실제',
            font=dict(size=label_size),  # 전체 폰트 크기 증가
            xaxis_title_font=dict(size=label_size),
            yaxis_title_font=dict(size=label_size),
            xaxis=dict(tickfont=dict(size=label_size)),  # x축 레이블 크기 조정
            yaxis=dict(tickfont=dict(size=label_size)),   # y축 레이블 크기 조정
            height=500,  # 높이 증가
            width=600,   # 너비 증가
        )
        
        st.plotly_chart(cm_fig)

        # Classification Report
        class_report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
        df_class_report = pd.DataFrame(class_report).transpose()
        st.markdown("**Classification Report:**")
        st.dataframe(df_class_report.style.format("{:.4f}"))
        
        # True Negative, False Positive, False Negative, True Positive
        tn, fp, fn, tp = cm.ravel()

        # 양성예측도 (Positive Predictive Value, PPV) = Precision
        ppv = tp / (tp + fp)

        # 음성예측도 (Negative Predictive Value, NPV)
        npv = tn / (tn + fn)

        # Performance Metrics
        metrics = {
            "민감도 (Sensitivity/Recall)": recall_score(y_true, y_pred),
            "특이도 (Specificity)": cm[0, 0] / (cm[0, 0] + cm[0, 1]),
            "정확도 (Accuracy)": accuracy_score(y_true, y_pred),
            "정밀도 (Precision)": precision_score(y_true, y_pred),
            "F1 Score": f1_score(y_true, y_pred),
            "양성예측도 (Positive Predictive Value, PPV)": ppv,
            "음성예측도 (Negative Predictive Value, NPV)": npv
        }

        for metric, value in metrics.items():
            st.write(f"**{metric}: {value:.4f}**")

        # ROC AUC
        roc_auc = roc_auc_score(y_true, y_score)
        st.markdown(f"**ROC AUC: {roc_auc:.4f}**")

        # ROC Curve
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (AUC = {roc_auc:.4f})'))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))

        # 주어진 threshold에 가장 가까운 값을 찾아 그리기
        idx = np.argmin(np.abs(thresholds - threshold))
        fig.add_trace(go.Scatter(x=[fpr[idx]], y=[tpr[idx]], 
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
        fig.add_trace(go.Histogram(x=y_score[y_true == 0], name='음성', opacity=0.7))
        fig.add_trace(go.Histogram(x=y_score[y_true == 1], name='양성', opacity=0.7))
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
        
        # Scatter plot for data distribution
        # 데이터 준비
        indices = list(range(len(y_score)))
        y_pred = (y_score > threshold).astype(int)
        # Scatter plot 생성
        fig = go.Figure()
        # Ground truth에 따라 색상 지정
        colors = ['blue' if gt == 0 else 'red' for gt in y_true]
        # 임계값에 따른 분류 결과에 따라 모양 지정
        symbols = ['circle' if pred == 0 else 'diamond' for pred in y_pred]

        # Scatter plot 추가
        fig.add_trace(go.Scatter(
            x=indices,
            y=y_score,
            mode='markers',
            marker=dict(
                color=colors,
                symbol=symbols,
                size=10
            ),
            text=[f'Score: {score:.2f}<br>Predicted: {"Negative" if pred == 0 else "Positive"}<br>Actual: {"Negative" if true == 0 else "Positive"}' 
                for score, pred, true in zip(y_score, y_pred, y_true)],
            hoverinfo='text'
        ))

        # 임계값 선 추가
        fig.add_hline(y=threshold, line_dash="dash", line_color="green", annotation_text="Threshold")

        # 레이아웃 설정
        fig.update_layout(
            title='데이터 분포 파악: Scores, Predictions, Threshold, and Ground Truth',
            xaxis_title='Data Point Index',
            yaxis_title='Score',
            showlegend=False
        )
        st.plotly_chart(fig)
        
        
        st.header("(부록)분석 결과에 대한 통계적 검정 결과")
        st.markdown("**Confusion Matrix 결과에 대한 통계검정**")

        tests = [
            accuracy_binomial_test(y_true, y_pred),
            auc_confidence_interval(y_true, y_score),
            confusion_matrix_chi2_test(y_true, y_pred),
            mcnemar_test(y_true, y_pred),
            cohen_kappa(y_true, y_pred),
            fishers_exact_test(y_true, y_pred)
        ]

        for test in tests:
            st.subheader(test['method'])
            st.write(test['description'])
            for key, value in test.items():
                if key not in ['method', 'description']:
                    st.write(f"{key}: {value}")
            st.write("---")

    except Exception as e:
        st.error(f"분석 중 오류가 발생했습니다: {str(e)}")
        st.write("오류 상세 정보:")
        st.write(e)
        st.write("데이터 및 선택한 컬럼을 다시 확인해주세요.")

def main():
    st.title('분류모델 성능 평가')
    
    # 세션 상태 초기화
    if 'file_uploaded' not in st.session_state:
        st.session_state.file_uploaded = False
    
    uploaded_file = st.file_uploader("CSV 또는 Excel 파일 업로드", type=['csv', 'xlsx', 'xls'])
    
    if uploaded_file is not None:
        # 파일이 새로 업로드되면 세션 상태를 리셋
        if st.session_state.file_uploaded and uploaded_file.name != st.session_state.last_uploaded_file:
            st.session_state.clear()
            st.rerun()
        
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
                positive_label = st.selectbox("양성(뇌출혈 등) 레이블 선택", unique_values, index=0 if len(unique_values) > 1 else 0)
                
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
**요구사항 목록**  
  
**1. 파일 업로드**
 - AI 분석 결과와 정답이 포함된 csv, excel 등 파일을 사용자가 선택 -> 로드
 - 업로드한 파일명과 성공 여부 확인
 - 데이터를 파악할 수 있게 보여주기 (기능 숨김 디폴트)
    - 10개 row 표시
    - 컬럼 수와 컬럼 속성 표시
    - (미구현)데이터 차트 미리보기-컬럼 속성에 따라 적절한 plot 선택

**2. 분석을 위한 사전 설정**
 - 분석 대상 컬럼 선택
    - 자동으로 AI score와 정답 컬럼 추천
    - 사용자가 드롭다운에서 선택 가능하며 분석 시 확정 (컬럼 load)
 - 정답 컬럼의 양성 레이블 선택 (정답 컬럼 load)
 - 양성 판정 임계값 설정

**3. 분석**
 - 위 설정한 정보로 분석 시도
 - 분석 실패 시 재시도
 - Confusion Matrix
 - Classification Report
 - 예측 스코어 분포
 - 예측, threshold, ground truth

**4. 통계검정**
 - 다양한 검정 결과 표시
 
**5. 그 외**
 - (미구현)사전에 분석에 사용할 데이터 수를 순차적으로 또는 랜덤으로 선택하도록 함. (원하는 성능을 내도록 조정이 필요한 경우)

 - (미구현)데이터 파악 시 컬럼 속성에 따른 디스플레이 방법은 chatGPT 의존하거나, 룰베이스로 자동 시각화하도록 구성. -> chatGPT 의존도 낮춰야할 것

 - (미구현)자동 레이블링-필요시
    - 이제 이것도 한 번에 던져서 결과를 json으로 받을 수 있겠다. 한 건씩 하는게 아니라.
"""