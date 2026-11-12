"""Design tokens and styling for native Streamlit elements only."""
import streamlit as st

TEAL='#0b7467'
ORANGE='#dc8d53'
MINT='#8bcbbd'
INK='#202e35'
BACKGROUND='#f5f7f8'


def apply_theme():
    st.markdown('''<style>
    :root{--kontur-teal:#0b7467;--kontur-ink:#202e35;--kontur-muted:#7b878d;--kontur-line:#e7ebed;}
    [data-testid="stAppViewContainer"]{background:#f5f7f8;color:#202e35;}
    header[data-testid="stHeader"]{background:transparent;}
    [data-testid="stMainBlockContainer"]{max-width:1500px;padding:24px 38px 40px;}
    [data-testid="stToolbar"],#MainMenu,footer{display:none;}
    h1,h2,h3{letter-spacing:-.035em;color:#202e35;}
    h1{font-weight:700;line-height:1.08!important;}
    h2{font-size:1.3rem!important;font-weight:650!important;}
    h3{font-size:1.05rem!important;font-weight:600!important;}
    [data-testid="stCaptionContainer"]{color:#7b878d;}
    [data-testid="stVerticalBlockBorderWrapper"]>[data-testid="stVerticalBlock"]{background:#fff;}
    [data-testid="stVerticalBlockBorderWrapper"]{border-radius:14px;border-color:#e7ebed!important;}
    [data-testid="stMetric"]{background:#fff;border:1px solid #e7ebed;border-radius:12px;padding:18px 20px;}
    [data-testid="stMetricLabel"]{color:#75837d;font-size:.78rem;}
    [data-testid="stMetricValue"]{font-size:1.9rem;font-weight:650;color:#0b7467;letter-spacing:-.05em;}
    [data-testid="stMetricDelta"]{font-size:.75rem;}
    [data-testid="stButton"] button,[data-testid="stDownloadButton"] button{border-radius:9px;font-size:.85rem;min-height:39px;}
    [data-testid="stBaseButton-primary"],[data-testid="stBaseButton-primaryFormSubmit"]{background:#0b7467;border-color:#0b7467;}
    [data-testid="stForm"]{border:1px solid #e7ebed!important;background:#fff;border-radius:14px;padding:20px;}
    [data-testid="stExpander"]{background:#fff;border-color:#e7ebed;border-radius:12px;}
    [data-testid="stDataFrame"]{border-radius:10px;}
    [data-baseweb="input"],[data-baseweb="select"]>div{border-radius:8px;background:#fff;}
    .st-key-native_brand{border-bottom:1px solid #e7ebed;padding-bottom:14px;margin-bottom:2px;}
    .st-key-native_brand h2{font-size:1.6rem!important;font-weight:800!important;letter-spacing:-.07em;}
    .st-key-native_page{margin:0 0 16px;}
    .st-key-native_page button{border-radius:8px;}
    .st-key-native_footer{border-top:1px solid #e7ebed;padding-top:16px;margin-top:24px;}
    [data-testid="stAlert"]{border-radius:10px;}
    @media(max-width:700px){[data-testid="stMainBlockContainer"]{padding:14px 16px 30px;}h1{font-size:2rem!important;}[data-testid="stMetricValue"]{font-size:1.5rem;}}
    @media(prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important;}}
    </style>''',unsafe_allow_html=True)


def render_brand():
    with st.container(key='native_brand'):
        left,right=st.columns([3,2],vertical_alignment='center')
        with left: st.markdown('## ✳ контур')
        with right: st.caption('Лаборатория спроса · Python / Streamlit')


def render_footer():
    with st.container(key='native_footer'):
        st.caption('КОНТУР · Сценарная модель спроса на авиабилеты. Наблюдаемые данные и модельные оценки обозначены отдельно.')
