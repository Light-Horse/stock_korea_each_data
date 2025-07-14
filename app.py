import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pykrx import stock
import time
from datetime import datetime

# --- 기본 설정 ---
# Matplotlib 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
# pandas 출력 옵션
pd.set_option('display.max_columns', None)


# --- 데이터 처리 및 분석 함수 ---

def calculate_ema(values, days):
    """지수이동평균(EMA)을 계산합니다."""
    return values.ewm(span=days, adjust=False).mean()

# @st.cache_data: 함수의 결과를 캐시합니다. 동일한 종목코드로 함수가 다시 호출되면
# 네트워크 요청 없이 저장된 데이터를 즉시 반환하여 속도를 높입니다.
@st.cache_data
def get_stock_data_for_app(stock_code):
    """
    pykrx를 통해 주식 데이터를 직접 가져와 데이터프레임으로 반환합니다.
    (사용자가 제공한 DB 저장 스크립트의 로직을 그대로 사용)
    """
    # 약 6개월 전부터 오늘까지의 데이터를 넉넉하게 조회
    start_date = (pd.Timestamp.now() - pd.DateOffset(months=6)).strftime('%Y%m%d')
    end_date = datetime.now().strftime("%Y%m%d")

    # 1. 종가 및 시가총액 데이터 조회
    ohlcv = stock.get_market_ohlcv(start_date, end_date, stock_code)
    market_cap_origin = stock.get_market_cap(start_date, end_date, stock_code)
    market_cap = market_cap_origin[['시가총액']]

    # 2. 기관 및 외인 매매 데이터 조회
    inst_foreign = stock.get_market_trading_value_by_date(start_date, end_date, stock_code)
    inst_foreign['기관 순매수'] = inst_foreign['기관합계']
    inst_foreign['외인 순매수'] = inst_foreign['외국인합계']

    # 3. 데이터 병합
    df = ohlcv.join(inst_foreign[['기관 순매수', '외인 순매수']]).join(market_cap)

    # 4. 5일 합계 계산
    df['기관 순매수 5일 합계'] = df['기관 순매수'].rolling(window=5).sum()
    df['외인 순매수 5일 합계'] = df['외인 순매수'].rolling(window=5).sum()
    
    # 최근 77개의 유효한 데이터만 선택
    return df.dropna().tail(77)

def process_and_analyze_data(stock_code):
    """가져온 데이터를 기반으로 최종 분석 지표를 계산합니다."""
    data = get_stock_data_for_app(stock_code)
    if data.empty:
        return None, None

    # '시기외' 지표 계산
    data['시기외'] = (data['기관 순매수 5일 합계'] + data['외인 순매수 5일 합계']) / data['시가총액'] * 100

    # MACD 관련 지표 계산
    data['시기외12'] = calculate_ema(data['시기외'], 12)
    data['시기외26'] = calculate_ema(data['시기외'], 26)
    data['MACD'] = data['시기외12'] - data['시기외26']
    data['시그널'] = calculate_ema(data['MACD'], 9)
    data['MACD 오실레이터'] = data['MACD'] - data['시그널']
    
    # 통계치 계산
    stats = {
        '상위 10%': data['MACD 오실레이터'].quantile(0.9),
        '상위 25%': data['MACD 오실레이터'].quantile(0.75),
        '평균': data['MACD 오실레이터'].mean(),
        '하위 25%': data['MACD 오실레이터'].quantile(0.25),
        '하위 10%': data['MACD 오실레이터'].quantile(0.1),
    }
    return data, stats

def create_macd_graph(data, stats, stock_name, stock_code):
    """MACD 오실레이터 그래프를 생성합니다."""
    fig, ax1 = plt.subplots(figsize=(14, 10))
    ax1.plot(data.index, data['시가총액'], label='시가총액', color='black')
    ax1.set_ylabel('시가총액', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, axis='y', linestyle=':')
    ax2 = ax1.twinx()
    ax2.plot(data.index, data['MACD 오실레이터'], label='MACD 오실레이터', color='red')
    ax2.set_ylabel('MACD 오실레이터', color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.axhline(0, color='gray', linestyle='--', linewidth=0.7)
    for key, color, style in [('상위 10%', 'green', '--'), ('상위 25%', 'blue', '--'), ('평균', 'purple', '-'), ('하위 25%', 'blue', '--'), ('하위 10%', 'green', '--')]:
        value = stats[key]
        ax2.axhline(value, color=color, linestyle=style, linewidth=1.2, label=f"{key} ({value:.2f})")
    for date in data.index:
        ax1.axvline(date, color='gray', linestyle=':', linewidth=0.5)
    plt.title(f'{stock_name}({stock_code}) - 시가총액과 MACD 오실레이터', fontsize=16)
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    plt.tight_layout()
    return fig

@st.cache_data
def search_stock_code(keyword):
    """종목명을 기반으로 종목 코드를 검색합니다."""
    # 오늘 날짜를 기준으로 가장 최신 종목 목록을 가져오도록 수정
    today_str = datetime.now().strftime("%Y%m%d")
    tickers = stock.get_market_ticker_list(date=today_str, market="ALL")
    
    stock_info = {stock.get_market_ticker_name(ticker): ticker for ticker in tickers}
    if keyword in stock_info:
        return {keyword: stock_info[keyword]}, None
    similar_stocks = {name: code for name, code in stock_info.items() if keyword in name}
    return (None, similar_stocks) if similar_stocks else (None, None)

# --- Streamlit 앱 인터페이스 ---

st.title('📈 MACD 오실레이터를 활용한 주식 분석')

if 'selected_stock_code' not in st.session_state:
    st.session_state.selected_stock_code = None
    st.session_state.selected_stock_name = None
    st.session_state.similar_stocks = None

stock_name_input = st.text_input('분석할 종목명을 입력하세요 (예: 삼성전자):', '')

if st.button('종목 검색'):
    if stock_name_input:
        exact_match, similar_stocks = search_stock_code(stock_name_input)
        if exact_match:
            st.session_state.selected_stock_name = list(exact_match.keys())[0]
            st.session_state.selected_stock_code = list(exact_match.values())[0]
            st.session_state.similar_stocks = None
            st.rerun()
        elif similar_stocks:
            st.warning("정확히 일치하는 종목이 없습니다. 아래 목록에서 선택해주세요.")
            st.session_state.similar_stocks = similar_stocks
            st.session_state.selected_stock_code = None
        else:
            st.error("해당 이름으로 종목을 찾을 수 없습니다.")
            st.session_state.similar_stocks = None
            st.session_state.selected_stock_code = None
    else:
        st.info("종목명을 입력 후 검색 버튼을 눌러주세요.")

if st.session_state.similar_stocks:
    options = [f"{name} ({code})" for name, code in st.session_state.similar_stocks.items()]
    selected_option = st.radio("비슷한 종목 목록:", options)
    if st.button("이 종목으로 분석하기"):
        name, code = selected_option.rsplit(' (', 1)
        code = code[:-1]
        st.session_state.selected_stock_name = name
        st.session_state.selected_stock_code = code
        st.session_state.similar_stocks = None
        st.rerun()

if st.session_state.selected_stock_code:
    st.header(f"분석 결과: {st.session_state.selected_stock_name} ({st.session_state.selected_stock_code})")
    with st.spinner('데이터를 불러와 분석하는 중입니다...'):
        data, stats = process_and_analyze_data(st.session_state.selected_stock_code)
    
    if data is not None and not data.empty:
        st.subheader("📊 MACD 오실레이터 통계")
        col1, col2, col3 = st.columns(3)
        col1.metric("상위 10%", f"{stats['상위 10%']:.4f}")
        col2.metric("상위 25%", f"{stats['상위 25%']:.4f}")
        col3.metric("평균", f"{stats['평균']:.4f}")
        col1.metric("하위 25%", f"{stats['하위 25%']:.4f}", delta_color="inverse")
        col2.metric("하위 10%", f"{stats['하위 10%']:.4f}", delta_color="inverse")

        st.subheader("📈 그래프")
        graph = create_macd_graph(data, stats, st.session_state.selected_stock_name, st.session_state.selected_stock_code)
        st.pyplot(graph)

        with st.expander("최근 10일 데이터 보기"):
            st.dataframe(data.sort_index(ascending=False).head(10))
    else:
        st.error("데이터를 가져오거나 처리하는 데 실패했습니다. 종목 코드를 확인해주세요.")
