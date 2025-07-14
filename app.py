import streamlit as st
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
# import base64 # Streamlit에서는 직접 이미지를 다루므로 필요 없습니다.
from pykrx import stock
import time

# Matplotlib 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 부호 깨짐 방지


# --- 함수 (기존 Flask 앱과 대부분 동일) ---

def calculate_ema(values, days):
    """지수이동평균(EMA)을 계산합니다."""
    return values.ewm(span=days, adjust=False).mean()


def load_from_db(stock_code, db_name="stock_data2.db"):
    """데이터베이스에서 주식 데이터를 불러옵니다."""
    conn = sqlite3.connect(db_name)
    table_name = f"stock_{stock_code}"
    query = f"SELECT * FROM {table_name} ORDER BY 날짜 DESC LIMIT 77"
    try:
        df = pd.read_sql(query, conn, index_col="날짜", parse_dates=["날짜"])
    except pd.io.sql.DatabaseError:
        # 데이터베이스에 테이블이 없을 경우 에러 메시지를 표시합니다.
        st.error(f"오류: 데이터베이스에서 '{table_name}' 테이블을 찾을 수 없습니다. 종목 코드와 데이터베이스를 확인해주세요.")
        return pd.DataFrame()
    finally:
        conn.close()
    return df.sort_index()


def process_data(stock_code):
    """불러온 데이터를 가공하여 MACD 및 다른 지표들을 계산합니다."""
    data = load_from_db(stock_code)
    if data.empty:
        return None

    # 시가총액 대비 기관/외국인 순매수 비율 계산
    data['시기외'] = (data['기관 순매수 5일 합계'] + data['외인 순매수 5일 합계']) / data['시가총액'] * 100
    data['시기외12'] = calculate_ema(data['시기외'], 12)
    data['시기외26'] = calculate_ema(data['시기외'], 26)

    # MACD 관련 지표 계산
    data['MACD'] = data['시기외12'] - data['시기외26']
    data['시그널'] = calculate_ema(data['MACD'], 9)
    data['MACD 오실레이터'] = data['MACD'] - data['시그널']
    return data


def calculate_stats(data):
    """MACD 오실레이터의 통계치를 계산합니다."""
    stats = {
        '상위 10%': data['MACD 오실레이터'].quantile(0.9),
        '상위 25%': data['MACD 오실레이터'].quantile(0.75),
        '평균': data['MACD 오실레이터'].mean(),
        '하위 25%': data['MACD 오실레이터'].quantile(0.25),
        '하위 10%': data['MACD 오실레이터'].quantile(0.1),
    }
    return stats


def create_macd_graph(data, stats, stock_name, stock_code):
    """MACD 오실레이터 그래프를 생성합니다."""
    fig, ax1 = plt.subplots(figsize=(14, 10))

    # 시가총액 그래프
    ax1.plot(data.index, data['시가총액'], label='시가총액', color='black')
    ax1.set_ylabel('시가총액', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, axis='y', linestyle=':')

    # MACD 오실레이터 그래프
    ax2 = ax1.twinx()
    ax2.plot(data.index, data['MACD 오실레이터'], label='MACD 오실레이터', color='red')
    ax2.set_ylabel('MACD 오실레이터', color='black')
    ax2.tick_params(axis='y', labelcolor='black')

    # 통계 기준선 추가
    ax2.axhline(0, color='gray', linestyle='--', linewidth=0.7)
    ax2.axhline(stats['상위 10%'], color='green', linestyle='--', linewidth=1, label=f"상위 10% ({stats['상위 10%']:.2f})")
    ax2.axhline(stats['상위 25%'], color='blue', linestyle='--', linewidth=1, label=f"상위 25% ({stats['상위 25%']:.2f})")
    ax2.axhline(stats['평균'], color='purple', linestyle='-', linewidth=1.5, label=f"평균 ({stats['평균']:.2f})")
    ax2.axhline(stats['하위 25%'], color='blue', linestyle='--', linewidth=1, label=f"하위 25% ({stats['하위 25%']:.2f})")
    ax2.axhline(stats['하위 10%'], color='green', linestyle='--', linewidth=1, label=f"하위 10% ({stats['하위 10%']:.2f})")

    # 세로 점선 추가 (날짜 구분)
    for date in data.index:
        ax1.axvline(date, color='gray', linestyle=':', linewidth=0.5)

    # 그래프 마무리
    plt.title(f'{stock_name}({stock_code}) - 시가총액과 MACD 오실레이터', fontsize=16)
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    plt.tight_layout()

    return fig


# @st.cache_data: 한 번 실행된 함수의 결과를 캐싱하여, 동일한 입력에 대해 함수를 다시 실행하지 않습니다.
# 이렇게 하면 앱의 성능이 크게 향상됩니다.
@st.cache_data
def search_stock_code(keyword):
    """종목명을 기반으로 종목 코드를 검색합니다."""
    # pykrx에서 모든 종목의 티커와 이름을 가져옵니다.
    tickers = stock.get_market_ticker_list(market="ALL")
    stock_info = {stock.get_market_ticker_name(ticker): ticker for ticker in tickers}

    # 정확히 일치하는 경우
    if keyword in stock_info:
        return {keyword: stock_info[keyword]}, None

    # 키워드를 포함하는 비슷한 종목 검색
    similar_stocks = {name: code for name, code in stock_info.items() if keyword in name}

    if similar_stocks:
        return None, similar_stocks

    return None, None


# --- Streamlit 앱 인터페이스 ---

st.title('📈 MACD 오실레이터를 활용한 주식 분석')

# st.session_state를 사용하여 사용자의 선택을 기억합니다.
if 'selected_stock_code' not in st.session_state:
    st.session_state.selected_stock_code = None
    st.session_state.selected_stock_name = None
    st.session_state.similar_stocks = None

# 종목명 입력 필드
stock_name_input = st.text_input('분석할 종목명을 입력하세요 (예: 삼성전자):', '')

# 검색 버튼
if st.button('종목 검색'):
    if stock_name_input:
        # 입력된 종목명으로 검색 실행
        exact_match, similar_stocks = search_stock_code(stock_name_input)

        if exact_match:
            # 정확히 일치하는 종목이 있으면 바로 선택
            st.session_state.selected_stock_name = list(exact_match.keys())[0]
            st.session_state.selected_stock_code = list(exact_match.values())[0]
            st.session_state.similar_stocks = None  # 비슷한 종목 리스트는 초기화
            st.rerun()  # 화면을 새로고침하여 바로 분석 결과를 보여줍니다.
        elif similar_stocks:
            # 비슷한 종목이 있으면 사용자에게 선택지를 제공
            st.warning("정확히 일치하는 종목이 없습니다. 아래 목록에서 선택해주세요.")
            st.session_state.similar_stocks = similar_stocks
            st.session_state.selected_stock_code = None  # 이전 선택 초기화
        else:
            # 검색 결과가 없으면 에러 메시지 표시
            st.error("해당 이름으로 종목을 찾을 수 없습니다.")
            st.session_state.similar_stocks = None
            st.session_state.selected_stock_code = None
    else:
        st.info("종목명을 입력 후 검색 버튼을 눌러주세요.")

# 비슷한 종목 리스트가 있는 경우, 라디오 버튼으로 선택지를 제공
if st.session_state.similar_stocks:
    # '종목명 (종목코드)' 형태로 선택지를 만듭니다.
    options = [f"{name} ({code})" for name, code in st.session_state.similar_stocks.items()]
    selected_option = st.radio("비슷한 종목 목록:", options)

    if st.button("이 종목으로 분석하기"):
        # 사용자가 선택한 종목의 이름과 코드를 분리합니다.
        name, code = selected_option.rsplit(' (', 1)
        code = code[:-1]  # 마지막 ')' 문자 제거

        st.session_state.selected_stock_name = name
        st.session_state.selected_stock_code = code
        st.session_state.similar_stocks = None  # 선택 후 목록 초기화
        st.rerun()  # 화면을 새로고침하여 분석 결과를 보여줍니다.

# 종목이 최종 선택되었을 때 분석 결과를 표시
if st.session_state.selected_stock_code:
    st.header(f"분석 결과: {st.session_state.selected_stock_name} ({st.session_state.selected_stock_code})")

    # 데이터를 불러오고 처리하는 동안 스피너(로딩 애니메이션)를 표시
    with st.spinner('데이터를 불러와 처리하는 중입니다...'):
        data = process_data(st.session_state.selected_stock_code)

    if data is not None and not data.empty:
        stats = calculate_stats(data)

        # 통계치를 보기 좋게 컬럼으로 나누어 표시
        st.subheader("📊 MACD 오실레이터 통계")
        col1, col2, col3 = st.columns(3)
        col1.metric("상위 10%", f"{stats['상위 10%']:.4f}")
        col2.metric("상위 25%", f"{stats['상위 25%']:.4f}")
        col3.metric("평균", f"{stats['평균']:.4f}")
        col1.metric("하위 25%", f"{stats['하위 25%']:.4f}", delta_color="inverse")
        col2.metric("하위 10%", f"{stats['하위 10%']:.4f}", delta_color="inverse")

        # 그래프 생성 및 표시
        st.subheader("📈 그래프")
        graph = create_macd_graph(data, stats, st.session_state.selected_stock_name,
                                  st.session_state.selected_stock_code)
        st.pyplot(graph)

        # 원본 데이터를 확장/축소 가능한 섹션 안에 표시
        with st.expander("최근 10일 데이터 보기"):
            st.dataframe(data.sort_index(ascending=False).head(10))
    else:
        # 데이터 처리 중 오류가 발생했음을 알림
        st.error("선택된 종목의 데이터를 가져오거나 처리하는 데 실패했습니다.")
