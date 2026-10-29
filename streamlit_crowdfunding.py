"""모금나침반 Streamlit 앱 · 노트북과 역할 분리
담당: 저장 모델 로드, 조건 입력, 성공·실패 예측, 조건 변경 비교 및 시각화.
학습/전처리 적합은 crowdfunding_analysis.ipynb가 수행합니다.
실행: python -m streamlit run streamlit_crowdfunding.py
"""
from pathlib import Path
from functools import lru_cache
import hashlib,io,json,math
import numpy as np
import pandas as pd
import joblib,sklearn

APP_DIR=Path(__file__).resolve().parent
FEATURES=['goal_native','currency','country','category_group','duration_days']
CATEGORIES={'film_video':'영화·영상','animation':'애니메이션','games':'게임','publishing':'출판','art':'미술','comics':'만화','crafts':'공예','dance':'무용','design':'디자인','journalism':'저널리즘','music':'음악','photography':'사진','theater':'공연·연극'}
COUNTRIES={'US':'미국','GB':'영국','JP':'일본','CA':'캐나다','AU':'호주','DE':'독일','FR':'프랑스','AT':'오스트리아','BE':'벨기에','CH':'스위스','DK':'덴마크','ES':'스페인','GR':'그리스','HK':'홍콩','IE':'아일랜드','IT':'이탈리아','LU':'룩셈부르크','MX':'멕시코','NL':'네덜란드','NO':'노르웨이','NZ':'뉴질랜드','PL':'폴란드','SE':'스웨덴','SG':'싱가포르','SI':'슬로베니아'}
NATIVE_CURRENCY={'US':'USD','GB':'GBP','JP':'JPY','CA':'CAD','AU':'AUD','CH':'CHF','DK':'DKK','HK':'HKD','MX':'MXN','NO':'NOK','NZ':'NZD','PL':'PLN','SE':'SEK','SG':'SGD'}
LABELS={0:'목표 미달 주의',1:'목표 달성 예상'}

def resource_bytes(name):
    if name.startswith('/') or '..' in Path(name).parts:raise ValueError('잘못된 자료 경로입니다.')
    return (APP_DIR/name).read_bytes()

@lru_cache(maxsize=16)
def resource_json(name):return json.loads(resource_bytes(name).decode('utf-8-sig'))

def resource_csv(name):return pd.read_csv(io.BytesIO(resource_bytes(name)))
def model_spec():return resource_json('assets/model_spec.json')
def evaluation():return resource_json('assets/test_metrics.json')

@lru_cache(maxsize=1)
def load_model():
    spec=model_spec();binary=resource_bytes('models/random_forest.joblib')
    if hashlib.sha256(binary).hexdigest()!=spec['model_sha256']:raise RuntimeError('모델 파일과 학습 기록이 일치하지 않습니다. 앱 전체 폴더를 다시 확인해 주세요.')
    if sklearn.__version__!=spec['versions']['sklearn']:raise RuntimeError('학습 때와 실행 환경이 다릅니다. 제공한 requirements.txt로 실행 환경을 설치해 주세요.')
    model=joblib.load(io.BytesIO(binary))
    if model.named_steps['classifier'].__class__.__name__!='RandomForestClassifier':raise RuntimeError('랜덤포레스트 모델이 아닙니다.')
    if list(model.feature_names_in_)!=FEATURES:raise RuntimeError('모델 입력변수의 순서가 다릅니다.')
    # 예측 값에는 영향 없는 병렬 실행 수. 소규모 웹 입력에 과도한 작업 스레드를 만들지 않는다.
    model.set_params(classifier__n_jobs=1)
    return model

def currencies_for(country):
    codes=sorted(c for co,c in model_spec()['observed_country_currency_pairs'] if co==country)
    native=NATIVE_CURRENCY.get(country,'EUR')
    return sorted(codes,key=lambda c:(c!=native,c))

def validate_native(goal,currency,country,category,days):
    try:goal=float(goal);days=float(days)
    except (ValueError,TypeError):raise ValueError('목표금액과 기간을 숫자로 입력해 주세요.')
    if not math.isfinite(goal) or goal<=0:raise ValueError('목표금액은 0보다 큰 금액이어야 합니다.')
    if not math.isfinite(days):raise ValueError('모금 기간을 올바르게 입력해 주세요.')
    spec=model_spec()
    if category not in spec['known_categories']:raise ValueError('지원하는 13개 창작 분야 중에서 선택해 주세요.')
    if [country,currency] not in spec['observed_country_currency_pairs']:raise ValueError('학습자료에 없는 국가·통화 조합입니다. 원화는 원화 계획 비교 모드를 사용해 주세요.')
    lo,hi=spec['training_duration_range']
    if not lo<=days<=hi:raise ValueError('모금 기간이 학습 범위인 약 1~60일을 벗어났습니다.')
    bounds=spec['training_goal_ranges_by_currency'][currency]
    if not bounds['min']<=goal<=bounds['max']:
        raise ValueError(f"환산 후 목표금액은 {currency} {bounds['min']:,.0f}~{bounds['max']:,.0f}의 학습 범위여야 합니다.")
    return dict(goal_native=goal,currency=currency,country=country,category_group=category,duration_days=days)

def analyze(category,goal,days,mode='krw',fx=1400.,country='US',currency='USD',candidate_id=None):
    if mode not in ['krw','native']:raise ValueError('입력 방식을 확인해 주세요.')
    try:goal=float(goal)
    except (ValueError,TypeError):raise ValueError('목표금액을 숫자로 입력해 주세요.')
    if not math.isfinite(goal) or goal<=0:raise ValueError('목표금액은 0보다 큰 금액이어야 합니다.')
    original_goal=goal
    if mode=='krw':
        if not goal.is_integer():raise ValueError('원화 목표금액은 정수로 입력해 주세요.')
        try:fx=float(fx)
        except (ValueError,TypeError):raise ValueError('가정 환율을 숫자로 입력해 주세요.')
        if not math.isfinite(fx) or not 100<=fx<=10000:raise ValueError('가정 환율은 1 USD당 100~10,000원 범위로 입력해 주세요.')
        goal=goal/fx;country='US';currency='USD'
    inputs=validate_native(goal,currency,country,category,days)
    frame=pd.DataFrame([inputs],columns=FEATURES);model=load_model() if candidate_id is None else load_variant(candidate_id)
    prob=float(model.predict_proba(frame)[0,list(model.classes_).index(1)])
    pred=int(model.predict(frame)[0])
    return dict(model='RandomForestClassifier',model_sha256=model_spec()['model_sha256'] if candidate_id is None else variant_record(candidate_id)['sha256'],candidate_id=candidate_id or model_spec()['selected_candidate'],run_id=model_spec()['run_id'],input_mode=mode,
        entered_goal=original_goal,entered_currency='KRW' if mode=='krw' else currency,
        assumed_krw_per_usd=fx if mode=='krw' else None,model_inputs=inputs,
        predicted_success=pred,predicted_label=LABELS[pred],success_score=prob,failure_score=1-prob,
        decision_rule='성공 점수 > 0.5이면 성공, 동점은 실패. sklearn Pipeline.predict 결과 사용.',
        score_note='학습된 분류모델의 미보정 점수이며 국내 실제 성공확률이 아님.',
        risk_grade=None,risk_note='미검증 3단계 등급 대신 이진 분류와 목표 미달 점수를 표시.',snapshot_date=model_spec()['snapshot_date'])

def scenarios(result):
    x=result['model_inputs'];goal=result['entered_goal'];days=x['duration_days'];mode=result['input_mode']
    plans=[('목표 20% 축소',goal*.8,days),('현재 계획',goal,days),('목표 20% 확대',goal*1.2,days),('기간 7일 단축',goal,days-7),('기간 7일 연장',goal,days+7)]
    rows=[]
    for label,g,d in plans:
        g=round(g) if mode=='krw' else round(g,2)
        row={'계획':label,'목표금액':g,'통화':result['entered_currency'],'기간(일)':d,'성공 점수':None,'점수 차이(%p)':None,'분류':'','안내':''}
        try:
            if not 1<=d<=60:raise ValueError('앱의 비교 범위 1~60일을 벗어납니다.')
            r=analyze(x['category_group'],g,d,mode,result['assumed_krw_per_usd'],x['country'],x['currency'])
            row.update({'성공 점수':r['success_score'],'점수 차이(%p)':(r['success_score']-result['success_score'])*100,'분류':r['predicted_label']})
        except ValueError as e:row['안내']=str(e);row['분류']='비교 범위 밖'
        rows.append(row)
    return pd.DataFrame(rows)

def domestic_cases():return resource_csv('assets/domestic_cases.csv')
def filter_domestic(group='전체',outcome='전체',query=''):
    d=domestic_cases()
    if group!='전체':d=d[d.category_group.eq(group)]
    if outcome!='전체':d=d[d.outcome_label.eq(1 if outcome=='성공' else 0)]
    if query.strip():d=d[d.title.fillna('').str.contains(query.strip(),case=False,regex=False)]
    return d.copy()

def check_runtime():
    model=load_model();r=analyze('film_video',3000000,30)
    return dict(model=model.named_steps['classifier'].__class__.__name__,model_sha256=model_spec()['model_sha256'],
        native_features=list(model.feature_names_in_),domestic_rows=len(domestic_cases()),example=r)


# 하이퍼파라미터 시뮬레이션: 노트북 8~9단계에서 학습·저장한 후보만 로드한다.
def variant_record(candidate_id):
    records=resource_json('assets/model_variants.json')['candidates']
    match=[r for r in records if r['candidate_id']==candidate_id]
    if len(match)!=1:raise ValueError('저장되지 않은 모델 설정입니다.')
    return match[0]

@lru_cache(maxsize=8)
def load_variant(candidate_id):
    record=variant_record(candidate_id)
    binary=resource_bytes(record['path'])
    if hashlib.sha256(binary).hexdigest()!=record['sha256']:raise RuntimeError('후보 모델과 기록이 일치하지 않습니다.')
    model=joblib.load(io.BytesIO(binary))
    assert model.named_steps['classifier'].__class__.__name__=='RandomForestClassifier'
    model.set_params(classifier__n_jobs=1)
    return model


import sys
import streamlit as st

CSS = """<style>
.stApp{background:#ffffff;color:#253345}
.block-container{max-width:1050px;padding-top:3.2rem;padding-bottom:2rem}
[data-testid="stSidebar"]{background:#f3f7fa}
[data-testid="stSidebar"] .block-container{padding-top:2rem}
h1{font-size:2rem!important;letter-spacing:-.05rem}h2,h3{letter-spacing:-.02rem}
[data-testid="stMetric"]{border:1px solid #e1e8ed;border-radius:10px;padding:16px 20px;background:white}
[data-testid="stMetricValue"]{color:#216b57}
.result-heading{font-size:13px;color:#627185;margin:0 0 6px}
.result-label{font-size:28px;font-weight:700;color:#216b57;margin:0 0 8px}
.result-label.caution{color:#a75b23}
.empty-result{padding:30px 0;color:#617084}
@media(max-width:700px){.block-container{padding-top:1.4rem}h1{font-size:1.55rem!important}}
</style>"""

def clear_result():
    st.session_state.pop('analysis_result', None)
    st.session_state.pop('analysis_error', None)

def input_sidebar():
    with st.sidebar:
        st.header('프로젝트 조건')
        category = st.selectbox('창작 분야', list(CATEGORIES), format_func=lambda v:CATEGORIES[v], key='category', on_change=clear_result)
        mode = st.radio('입력 기준', ['원화 계획 비교', '해외 프로젝트 분석'], key='input_mode', on_change=clear_result)
        if mode == '원화 계획 비교':
            goal = st.number_input('목표금액 (원)', min_value=0, value=3000000, step=100000, key='goal_krw', on_change=clear_result)
            country, currency, input_mode = 'US', 'USD', 'krw'
            with st.expander('환산 기준 설정'):
                fx = st.number_input('1 USD당 원화 (가정값)', min_value=0., value=1400., step=10., key='fx', on_change=clear_result)
                st.caption('1,400원은 현재 환율이 아닌 시연용 가정입니다. 미국·USD 기준으로 비교합니다.')
        else:
            country = st.selectbox('프로젝트 국가', list(COUNTRIES), format_func=lambda v:COUNTRIES[v], key='country', on_change=clear_result)
            choices = currencies_for(country)
            if st.session_state.get('currency') not in choices:
                st.session_state['currency'] = choices[0]
            currency = st.selectbox('목표금액의 통화', choices, key='currency', on_change=clear_result)
            goal = st.number_input(f'목표금액 ({currency})', min_value=0., value=5000., step=100., key='goal_native', on_change=clear_result)
            fx, input_mode = None, 'native'
        days = st.number_input('모금 기간 (일)', min_value=1, max_value=60, value=30, step=1, key='days', on_change=clear_result)
        if st.button('성공 여부 예측', type='primary', width='stretch', key='analyze'):
            clear_result()
            try:
                st.session_state['analysis_result'] = analyze(category, goal, days, input_mode, fx, country, currency)
            except ValueError as exc:
                st.session_state['analysis_error'] = str(exc)
        st.caption('입력값을 바꾸면 이전 결과가 초기화됩니다.')
        st.divider()
        st.caption('모금나침반 · 랜덤포레스트')
        st.caption('창작 분야 13종 · 해외 데이터 기반')

def prediction_view():
    result = st.session_state.get('analysis_result')
    if st.session_state.get('analysis_error'):
        st.error(st.session_state['analysis_error'])
    if not result:
        st.markdown('<div class="empty-result">왼쪽에서 분야·목표금액·모금 기간을 입력한 뒤<br><b>성공 여부 예측</b>을 눌러 주세요.</div>', unsafe_allow_html=True)
        st.info('해외 Kickstarter 자료로 학습한 모델입니다. 원화 입력은 미국·USD 기준의 참고 비교이며 국내 실제 성공확률을 뜻하지 않습니다.')
        return
    x = result['model_inputs']
    tone = '' if result['predicted_success'] else ' caution'
    st.markdown('<p class="result-heading">예측 결과</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="result-label{tone}">{result["predicted_label"]}</p>', unsafe_allow_html=True)
    st.caption(f"{CATEGORIES[x['category_group']]} · 목표금액 {result['entered_goal']:,.0f} {result['entered_currency']} · {x['duration_days']:g}일")
    a, b = st.columns(2)
    a.metric('성공 점수', f"{result['success_score']:.1%}")
    b.metric('목표 미달 점수', f"{result['failure_score']:.1%}")
    st.progress(result['success_score'])
    if result['input_mode'] == 'krw':
        st.caption(f"비교 기준: 미국 · {x['goal_native']:,.2f} USD · 가정 환율 1 USD = {result['assumed_krw_per_usd']:,.0f}원")
    st.info('해외 자료로 학습한 모델의 참고 점수입니다. 국내 실제 성공확률이나 프로젝트 성공을 보장하는 값은 아닙니다.')
    with st.expander('목표금액·기간을 바꾸어 비교'):
        rows = scenarios(result)
        st.dataframe(rows[['계획','목표금액','통화','기간(일)','성공 점수','분류','안내']], hide_index=True, width='stretch', column_config={
            '목표금액':st.column_config.NumberColumn(format='localized'),
            '성공 점수':st.column_config.ProgressColumn(min_value=0,max_value=1,format='percent')})
        st.caption('같은 모델에 입력 조건만 바꾼 결과입니다. 실제 성공률 개선 효과를 입증하지 않습니다.')
    with st.expander('예측 결과 안내'):
        st.write('성공은 모금 목표 달성, 실패는 목표 미달입니다. 성공 점수가 50%를 넘으면 성공으로 분류합니다. 점수가 같은 경우 실패로 분류합니다.')
        st.write('여러 결정트리의 결과를 종합하는 랜덤포레스트를 사용합니다. 점수는 실제 성공률에 맞춰 보정한 값이 아닙니다.')
        st.write('목표 미달 점수는 1에서 성공 점수를 뺀 값입니다. 작품의 완성도·제작 지연·배송·수익 위험은 평가하지 않습니다.')
    st.download_button('예측 결과 저장', json.dumps(result, ensure_ascii=False, indent=2).encode('utf-8'), 'mogum_prediction.json', 'application/json', key='download_result')

def main():
    st.set_page_config(page_title='모금나침반 · 크라우드펀딩 성공 예측', page_icon='🧭', layout='wide', initial_sidebar_state='expanded')
    st.markdown(CSS, unsafe_allow_html=True)
    st.title('창작 프로젝트 성공 예측')
    st.caption('랜덤포레스트 모델로 크라우드펀딩의 목표 달성 여부를 예측합니다.')
    try:
        load_model()
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        st.error(f'예측 모델을 불러오지 못했습니다. {exc}')
        st.stop()
    input_sidebar()
    prediction_view()

if __name__ == '__main__':
    if '--check' in sys.argv:
        print(json.dumps(check_runtime(), ensure_ascii=False, indent=2))
    else:
        main()
