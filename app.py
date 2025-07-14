import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pykrx import stock
from datetime import datetime
import platform
import os

# --- 폰트 설정 (가장 먼저 실행) ---
def set_font():
    os_name = platform.system()
    if os_name == "Linux":
        font_path = os.path.join("fonts", "NanumGothic.ttf")
        if os.path.exists(font_path):
            fm.fontManager.addfont(font_path)
            plt.rcParams['font.family'] = 'NanumGothic'
    elif os_name == "Windows":
        plt.rcParams['font.family'] = 'Malgun Gothic'
    elif os_name == "Darwin":
        plt.rcParams['font.family'] = 'AppleGothic'
    plt.rcParams['axes.unicode_minus'] = False

set_font()

# --- 데이터 처리 및 분석 함수 ---

def calculate_ema(values, days):
    return values.ewm(span=days, adjust=False).mean()

@st.cache_data
def get_stock_data_for_app(stock_code):
    start_date = (pd.Timestamp.now() - pd.DateOffset(months=6)).strftime('%Y%m%d')
    end_date = datetime.now().strftime("%Y%m%d")
    ohlcv = stock.get_market_ohlcv(start_date, end_date, stock_code)
    market_cap_origin = stock.get_market_cap(start_date, end_date, stock_code)
    market_cap = market_cap_origin[['시가총액']]
    inst_foreign = stock.get_market_trading_value_by_date(start_date, end_date, stock_code)
    inst_foreign['기관 순매수'] = inst_foreign['기관합계']
    inst_foreign['외인 순매수'] = inst_foreign['외국인합계']
    df = ohlcv.join(inst_foreign[['기관 순매수', '외인 순매수']]).join(market_cap)
    df['기관 순매수 5일 합계'] = df['기관 순매수'].rolling(window=5).sum()
    df['외인 순매수 5일 합계'] = df['외인 순매수'].rolling(window=5).sum()
    return df.dropna().tail(77)

def process_and_analyze_data(stock_code):
    data = get_stock_data_for_app(stock_code)
    if data.empty:
        return None, None
    data['시기외'] = (data['기관 순매수 5일 합계'] + data['외인 순매수 5일 합계']) / data['시가총액'] * 100
    data['시기외12'] = calculate_ema(data['시기외'], 12)
    data['시기외26'] = calculate_ema(data['시기외'], 26)
    data['MACD'] = data['시기외12'] - data['시기외26']
    data['시그널'] = calculate_ema(data['MACD'], 9)
    data['MACD 오실레이터'] = data['MACD'] - data['시그널']
    stats = {
        '상위 10%': data['MACD 오실레이터'].quantile(0.9), '상위 25%': data['MACD 오실레이터'].quantile(0.75),
        '평균': data['MACD 오실레이터'].mean(),
        '하위 25%': data['MACD 오실레이터'].quantile(0.25), '하위 10%': data['MACD 오실레이터'].quantile(0.1), # <--- 누락되었던 통계 추가
    }
    return data, stats

def create_macd_graph(data, stats, stock_name, stock_code):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(data.index, data['시가총액'], label='시가총액', color='black')
    ax1.set_ylabel('시가총액', color='black', fontsize=10)
    ax1.tick_params(axis='y', labelcolor='black', labelsize=8)
    ax1.grid(True, axis='y', linestyle=':')
    ax1.tick_params(axis='x', rotation=45, labelsize=8)

    ax2 = ax1.twinx()
    ax2.plot(data.index, data['MACD 오실레이터'], label='MACD 오실레이터', color='red')
    ax2.set_ylabel('MACD 오실레이터', color='black', fontsize=10)
    ax2.tick_params(axis='y', labelcolor='black', labelsize=8)
    ax2.axhline(0, color='gray', linestyle='--', linewidth=0.7)
    
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2_lines_labels = {lbl: hnd for hnd, lbl in zip(lines2, labels2)}

    # 범례 순서와 내용 정의 (하위 10%, 25% 다시 추가)
    legend_order = [
        ('상위 10%', 'green', '--'), ('상위 25%', 'blue', '--'), ('평균', 'purple', '-'),
        ('하위 25%', 'blue', '--'), ('하위 10%', 'green', '--')
    ]
    
    for key, color, style in legend_order:
        value = stats[key]
        line = ax2.axhline(value, color=color, linestyle=style, linewidth=1.2)
        ax2_lines_labels[f"{key} ({value:.2f})"] = line
        
    plt.title(f'{stock_name}({stock_code}) - 시가총액과 MACD 오실레이터', fontsize=14)
    
    # 범례를 그래프 하단 중앙에 5열로 배치
    fig.legend(lines + list(ax2_lines_labels.values()), ['시가총액', 'MACD 오실레이터'] + list(ax2_lines_labels.keys())[1:],
               loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=4, fontsize=9)
    
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    return fig

@st.cache_data
def search_stock_code(keyword):
    today_str = datetime.now().strftime("%Y%m%d")
    tickers = stock.get_market_ticker_list(date=today_str, market="ALL")
    stock_info = {stock.get_market_ticker_name(ticker): ticker for ticker in tickers}
    if keyword in stock_info:
        return {keyword: stock_info[keyword]}, None
    similar_stocks = {name: code for name, code in stock_info.items() if keyword in name}
    return (None, similar_stocks) if similar_stocks else (None, None)

# --- Streamlit 앱 인터페이스 ---

st.set_page_config(page_title="주식 분석", layout="centered")
st.title('📈 MACD 오실레이터를 활용한 주식 분석')

if 'selected_stock_code' not in st.session_state:
    st.session_state.update({'selected_stock_code': None, 'selected_stock_name': None, 'similar_stocks': None})

stock_name_input = st.text_input('분석할 종목명을 입력하세요 (예: 삼성전자):', '')

if st.button('종목 검색'):
    if stock_name_input:
        exact_match, similar_stocks = search_stock_code(stock_name_input)
        if exact_match:
            st.session_state.selected_stock_name, st.session_state.selected_stock_code = list(exact_match.items())[0]
            st.session_state.similar_stocks = None
            st.rerun()
        elif similar_stocks:
            st.warning("정확히 일치하는 종목이 없습니다. 아래 목록에서 선택해주세요.")
            st.session_state.similar_stocks = similar_stocks
        else:
            st.error("해당 이름으로 종목을 찾을 수 없습니다.")

if st.session_state.similar_stocks:
    options = [f"{name} ({code})" for name, code in st.session_state.similar_stocks.items()]
    selected_option = st.radio("비슷한 종목 목록:", options, horizontal=True)
    if st.button("이 종목으로 분석하기"):
        name, code = selected_option.rsplit(' (', 1)
        st.session_state.selected_stock_name = name
        st.session_state.selected_stock_code = code.replace(')', '')
        st.session_state.similar_stocks = None
        st.rerun()

if st.session_state.selected_stock_code:
    st.header(f"분석 결과: {st.session_state.selected_stock_name} ({st.session_state.selected_stock_code})")
    with st.spinner('데이터를 불러와 분석하는 중입니다...'):
        data, stats = process_and_analyze_data(st.session_state.selected_stock_code)
    
    if data is not None and not data.empty:
        st.subheader("📊 MACD 오실레이터 통계")
        # 누락되었던 '하위' 통계 다시 표시
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("상위 10%", f"{stats['상위 10%']:.4f}")
            st.metric("하위 10%", f"{stats['하위 10%']:.4f}", delta_color="inverse")
        with col2:
            st.metric("상위 25%", f"{stats['상위 25%']:.4f}")
            st.metric("하위 25%", f"{stats['하위 25%']:.4f}", delta_color="inverse")
        with col3:
            st.metric("평균", f"{stats['평균']:.4f}")

        st.subheader("📈 그래프")
        graph = create_macd_graph(data, stats, st.session_state.selected_stock_name, st.session_state.selected_stock_code)
        st.pyplot(graph)

        with st.expander("최근 10일 데이터 보기"):
            st.dataframe(data.sort_index(ascending=False).head(10))
    else:
        st.error("데이터를 가져오거나 처리하는 데 실패했습니다.")
