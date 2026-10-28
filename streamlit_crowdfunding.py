"""모금나침반 Streamlit 앱 · 노트북과 역할 분리
담당: 모델 로드, 프로토타이핑, 저장된 하이퍼파라미터 후보 시뮬레이션, 시각화.
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


import sys,json,html
import pandas as pd
import streamlit as st


CSS='''<style>
.stApp{background:#f6f8f5;color:#203a33}.block-container{max-width:1180px;padding-top:2.4rem;padding-bottom:2rem}
h1,h2,h3{color:#173f34!important}h1{letter-spacing:-1.4px}p{line-height:1.7}
.hero{background:linear-gradient(115deg,#123d32,#265f4a);border-radius:20px;padding:30px 34px;margin-bottom:22px;color:#fff;position:relative;overflow:hidden}
.hero:after{content:'✦';position:absolute;right:25px;top:-48px;color:#b7db9825;font-size:230px;line-height:1.4}
.hero .eyebrow{color:#b3d5bf;font-size:12px;letter-spacing:2px;font-weight:700}.hero h1{color:white!important;font-size:39px;margin:4px 0 6px}.hero p{color:#d9e8df;margin:0;font-size:16px;max-width:760px}.hero .tag{display:inline-block;margin-top:18px;border:1px solid #76a48c;border-radius:20px;padding:3px 12px;color:#e2eedf;font-size:12px}
div[data-testid="stMetric"]{background:#fff;border:1px solid #dde8df;padding:17px 19px;border-radius:13px}
div[data-testid="stMetricValue"]{color:#194f3e}div[data-testid="stVerticalBlockBorderWrapper"]>div{border-radius:14px}
div[data-testid="stTabs"] button{font-size:15px}.result-banner{border-radius:12px;padding:20px 24px;background:#e9f4ed;border:1px solid #c5dfce;margin:8px 0 18px}
.result-banner.alert{background:#fff2e5;border-color:#efd1af}.result-banner h3{margin:4px 0 7px;font-size:25px}.result-banner p{margin:0;font-size:14px}.kicker{font-size:12px;letter-spacing:1px;color:#526f61}
.placeholder{border:1px dashed #bdcfc1;border-radius:15px;padding:45px 26px;text-align:center;background:#fff;min-height:270px}.placeholder .symbol{font-size:42px;color:#487361}.placeholder p{color:#667c70}
.small-note{font-size:12px;color:#6c7e72}.section-number{color:#72987b;font-weight:700;font-size:12px;letter-spacing:1.5px}.case-card{background:white;border:1px solid #dbe5dd;border-radius:12px;padding:18px;margin-bottom:14px}.case-card a{font-size:16px;font-weight:650;color:#1c5843;text-decoration:none}.case-card p{font-size:13px;margin:6px 0;color:#63796c}.case-card .status{font-size:11px;color:#597b66;margin-bottom:8px}.case-card .money{color:#294d3a}
@media(max-width:700px){.block-container{padding:1rem}.hero{padding:23px}.hero h1{font-size:30px}.hero p{font-size:14px}.hero:after{font-size:150px;right:5px;top:-20px}.placeholder{min-height:180px}}
</style>'''

def clear_result():
    st.session_state.pop('analysis_result',None)
    st.session_state.pop('analysis_error',None)

def plan_page():
    st.markdown('#### 계획을 입력하고, 다른 조건과 비교해 보세요.')
    mode=st.radio('입력 방식',['원화 계획 비교','해외 프로젝트 분석'],horizontal=True,key='input_mode',on_change=clear_result)
    left,right=st.columns([1,1.12],gap='large')
    with left:
        with st.container(border=True):
            st.markdown('<span class="section-number">01 · 프로젝트 계획</span>',unsafe_allow_html=True)
            category=st.selectbox('창작 분야',list(CATEGORIES),format_func=lambda x:CATEGORIES[x],key='category',on_change=clear_result)
            if mode=='원화 계획 비교':
                goal=st.number_input('목표금액 (원)',min_value=0,value=3000000,step=100000,key='goal_krw',on_change=clear_result)
                country,currency='US','USD'
                with st.expander('환산 기준 확인',expanded=False):
                    fx=st.number_input('1 USD당 원화 (가정값)',min_value=0.,value=1400.,step=10.,key='fx',on_change=clear_result)
                    st.caption('1,400원은 현재 환율이 아닌 시연용 가정입니다. 원화 금액을 USD로 환산하고 미국 자료 기준으로 비교합니다.')
                if fx>0:st.caption(f'미국 기준 비교 금액: {goal/fx:,.2f} USD')
                st.caption('이 결과는 국내 프로젝트의 실제 성공확률이 아닙니다.')
                input_mode='krw'
            else:
                country=st.selectbox('프로젝트 국가',list(COUNTRIES),format_func=lambda x:COUNTRIES[x],key='country',on_change=clear_result)
                choices=currencies_for(country)
                if st.session_state.get('currency') not in choices:st.session_state['currency']=choices[0]
                currency=st.selectbox('목표금액의 통화',choices,key='currency',on_change=clear_result)
                goal=st.number_input(f'목표금액 ({currency})',min_value=0.,value=5000.,step=100.,key='goal_native',on_change=clear_result)
                fx=None;input_mode='native'
                bounds=model_spec()['training_goal_ranges_by_currency'][currency]
                st.caption(f"학습자료 금액 범위: {bounds['min']:,.0f}~{bounds['max']:,.0f} {currency}")
            days=st.number_input('모금 기간 (일)',min_value=1,max_value=60,value=30,step=1,key='days',on_change=clear_result)
            st.caption('앱 입력 범위는 1~60일입니다. 국가·통화·분야는 학습자료에 포함된 범위로 제공합니다.')
            clicked=st.button('계획 분석하기',type='primary',width='stretch',key='analyze')
            if clicked:
                clear_result()
                try:st.session_state['analysis_result']=analyze(category,goal,days,input_mode,fx,country,currency)
                except ValueError as e:st.session_state['analysis_error']=str(e)
            if st.session_state.get('analysis_error'):st.error(st.session_state['analysis_error'])
    with right:
        st.markdown('<span class="section-number">02 · 모델의 판단</span>',unsafe_allow_html=True)
        result=st.session_state.get('analysis_result')
        if not result:
            st.markdown('<div class="placeholder"><div class="symbol">✧</div><h3>내 프로젝트의 계획을 살펴보세요</h3><p>분야·금액·기간을 입력하면<br>목표 달성 예상 여부와 모델 점수를 보여드립니다.</p><p class="small-note">입력값을 바꾸면 이전 결과는 지워집니다.</p></div>',unsafe_allow_html=True)
        else:
            score=result['success_score'];x=result['model_inputs'];label=html.escape(result['predicted_label'])
            description='모델이 성공 쪽으로 분류했습니다. 제작비와 실행 계획도 함께 확인하세요.' if result['predicted_success'] else '모델이 목표 미달 쪽으로 분류했습니다. 목표금액과 실행 계획을 함께 검토하세요.'
            tone='' if result['predicted_success'] else ' alert'
            st.markdown(f'<div class="result-banner{tone}"><span class="kicker">랜덤포레스트 · 성공 / 실패 분류</span><h3>{label}</h3><p>{description}</p></div>',unsafe_allow_html=True)
            a,b=st.columns(2);a.metric('성공 점수',f'{score:.1%}',help='학습한 분류모델의 predict_proba 출력입니다. 실제 성공 발생률로 보정되지 않았습니다.')
            b.metric('목표 미달 점수',f'{result["failure_score"]:.1%}',help='1 − 성공 점수. 제작 지연이나 배송 사고의 위험을 측정한 값은 아닙니다.')
            st.progress(score)
            st.caption(f"{CATEGORIES[x['category_group']]} · {COUNTRIES[x['country']]} · {x['goal_native']:,.2f} {x['currency']} · {x['duration_days']:g}일")
            st.info('점수는 모델의 판단 정도입니다. 국내 성공확률로 해석하거나 성공을 보장하는 값으로 사용하지 마세요.')
            st.caption('위험 신호는 목표 달성 여부의 두 가지 분류 결과로 표시합니다. 이 모델은 별도 확률 보정을 수행하지 않았습니다.')
            st.download_button('분석 결과 저장',json.dumps(result,ensure_ascii=False,indent=2).encode('utf-8'),'mogum_rf_result.json','application/json',width='stretch',key='download_result')
    result=st.session_state.get('analysis_result')
    if result:
        st.divider();st.subheader('계획을 바꾸면 점수는 어떻게 달라질까요?')
        rows=scenarios(result)
        st.dataframe(rows,hide_index=True,width='stretch',column_config={'목표금액':st.column_config.NumberColumn(format='localized'),'성공 점수':st.column_config.ProgressColumn(min_value=0,max_value=1,format='percent'),'점수 차이(%p)':st.column_config.NumberColumn(format='%+.1f')})
        st.caption('같은 모델에 조건만 바꿔 입력한 비교입니다. 금액이나 기간을 변경하면 실제 성공률이 오른다는 인과효과를 입증하지 않습니다.')
    with st.expander('이 결과를 읽는 방법'):
        st.write('성공은 모금 목표를 달성한 경우, 실패는 목표를 달성하지 못한 경우입니다. 성공 점수가 50%를 넘으면 성공으로 분류하며, 동점이면 실패로 분류합니다.')
        st.write('성공/실패 비율이 불균형해 학습 과정에서 실패 사례의 비중을 높였습니다. 따라서 점수 70%가 실제 프로젝트 10개 중 7개 성공을 뜻하지 않습니다.')
        st.write('이 모델은 창작물의 완성도, 홍보 계획, 제작 지연이나 배송 완료를 직접 평가하지 않습니다.')

def domestic_page():
    cases=domestic_cases();total=len(cases);known=int(cases.goal_amount_krw.notna().sum())
    st.subheader('국내 프로젝트 100건에서 실제 사례를 찾아보세요')
    a,b,c=st.columns(3);a.metric('수집한 종료 사례',f'{total}건');b.metric('목표금액 확인',f'{known}건');c.metric('목표금액 미확인',f'{total-known}건')
    st.caption('성공 50건·목표 미달 50건을 찾아 모은 표본입니다. 국내 평균 성공률을 나타내지 않으며 모델 학습·성능 평가에는 사용하지 않았습니다.')
    a,b,c=st.columns([1.2,1,1.8])
    group=a.selectbox('사례 분야',['전체']+sorted(cases.category_group.unique()),key='case_group')
    outcome=b.selectbox('결과',['전체','성공','목표 미달'],key='case_outcome')
    query=c.text_input('제목 검색',placeholder='예: 영화, 책, 게임',key='case_query')
    filtered=filter_domestic(group,outcome,query)
    st.caption(f'검색 결과 {len(filtered)}건 · 공개자료 수집일 2026-10-28')
    if filtered.empty:st.info('조건에 맞는 사례가 없습니다. 검색어나 필터를 바꿔 보세요.')
    else:
        page=st.selectbox('보기 범위',range(1,(len(filtered)+11)//12+1),format_func=lambda p:f'{p}페이지',key=f'case_page_{group}_{outcome}_{query}')
        show=filtered.iloc[(page-1)*12:page*12]
        cols=st.columns(2)
        for i,(_,r) in enumerate(show.iterrows()):
            title=html.escape(str(r.title));url=html.escape(str(r.project_url),quote=True)
            amount=f'{r.goal_amount_krw:,.0f}원' if pd.notna(r.goal_amount_krw) else '미확인'
            status='성공' if r.outcome_label==1 else '목표 미달'
            with cols[i%2]:
                st.markdown(f'<div class="case-card"><div class="status">{html.escape(r.category_group)} · {status}</div><a href="{url}" target="_blank" rel="noopener noreferrer">{title} ↗</a><p class="money">목표 {amount} · 모금 {r.displayed_amount_krw:,.0f}원</p><p>{r.start_date} ~ {r.end_date} · {r.duration_days:g}일</p></div>',unsafe_allow_html=True)
    st.download_button('검색 결과 CSV 저장',filtered.to_csv(index=False).encode('utf-8-sig'),'tumblbug_cases_filtered.csv','text/csv',key='download_cases')
    st.caption('확인되지 않은 목표금액 35건은 비워 두었습니다. 달성률을 이용한 역산이나 추정값 보완은 하지 않았습니다. 모금 성공은 제작·배송 완료를 뜻하지 않습니다.')

def performance_page():
    result=evaluation();m=result['metrics'];cm=result['confusion_matrix'];baseline=result['baseline_all_success']
    st.subheader('새 모델의 실제 성능을 확인하세요')
    st.caption('랜덤포레스트 · 2026년 시험자료 7,490건 · 성공 6,749건 / 실패 741건')
    cols=st.columns(4)
    for col,label,value,help_text in [(cols[0],'정확도',f"{m['accuracy']:.1%}",'전체 프로젝트 중 성공/실패를 맞힌 비율'),(cols[1],'실패 정밀도',f"{m['precision_failure']:.1%}",'실패라고 경고한 사례 중 실제로 실패한 비율'),(cols[2],'실패 재현율',f"{m['recall_failure']:.1%}",'실제 실패한 사례 중 모델이 찾아낸 비율'),(cols[3],'평균 F1',f"{m['macro_f1']:.3f}",'성공 F1과 실패 F1의 산술평균')]:col.metric(label,value,help=help_text)
    st.warning(f"실패 {m['support_failure']:,}건 중 {cm[0][0]:,}건을 찾았지만, 실제 성공 {cm[1][0]:,}건에도 실패 경고를 했습니다. 높은 오경보가 현재 모델의 한계입니다.")
    left,right=st.columns([1,1.05],gap='large')
    with left:
        st.markdown('##### 실제 결과와 예측 결과')
        st.dataframe(pd.DataFrame(cm,index=['실제 실패','실제 성공'],columns=['실패 예측','성공 예측']),width='stretch')
        st.caption(f'실패 탐지 {cm[0][0]}건 · 실패 놓침 {cm[0][1]}건 · 성공 오경보 {cm[1][0]:,}건 · 성공 정답 {cm[1][1]:,}건')
    with right:
        st.markdown('##### 정확도 하나만 보지 않는 이유')
        st.write(f"모든 프로젝트를 성공이라고 답해도 정확도는 {baseline['accuracy']:.1%}입니다. 하지만 실패는 한 건도 찾지 못합니다.")
        st.write(f"우리 모델은 평균 F1을 {baseline['macro_f1']:.3f}에서 {m['macro_f1']:.3f}로 높였습니다. 정확도와 실패 탐지 성능을 함께 봐야 합니다.")
    st.markdown('##### 성공·실패별 평가')
    cls=pd.DataFrame([{'결과':label,'정밀도':m[f'precision_{key}'],'재현율':m[f'recall_{key}'],'F1':m[f'f1_{key}'],'표본수':m[f'support_{key}']} for label,key in [('실패','failure'),('성공','success')]])
    st.dataframe(cls,hide_index=True,width='stretch',column_config={c:st.column_config.NumberColumn(format='%.3f') for c in ['정밀도','재현율','F1']})
    st.markdown('##### 분야별로도 확인해 보세요')
    category=st.selectbox('평가할 분야',list(CATEGORIES),format_func=lambda x:CATEGORIES[x],key='performance_category')
    d=resource_csv('assets/category_metrics.csv');r=d[d.group.eq(category)].iloc[0]
    st.dataframe(pd.DataFrame([{'분야':CATEGORIES[category],'전체':int(r.rows),'실패':int(r.support_failure),'정확도':r.accuracy,'실패 정밀도':r.precision_failure,'실패 재현율':r.recall_failure,'평균 F1':r.macro_f1}]),hide_index=True,width='stretch',column_config={c:st.column_config.NumberColumn(format='%.3f') for c in ['정확도','실패 정밀도','실패 재현율','평균 F1']})
    if r.support_failure<30:st.caption('실패 표본이 30건 미만인 분야입니다. 이 분야의 성능 수치는 변동성이 클 수 있습니다.')
    with st.expander('학습·검증·시험을 어떻게 나눴나요?'):
        st.dataframe(resource_csv('assets/split_summary.csv'),hide_index=True,width='stretch')
        st.write('2020~2024년 자료로 학습하고 2025년 자료로 모델을 선택했습니다. 선택을 마친 뒤 2026년 자료로 시험했습니다. 같은 프로젝트가 분할 사이에 겹치지 않도록 정리했습니다.')
        st.write('이번 10단계 노트북에서 모델을 다시 학습했습니다. 과거 비교에서 확인한 2026년 시험자료를 재사용한 재현 실험이며, 새로운 독립 시험은 아닙니다. 시험 결과를 보고 이번 후보를 다시 선택하지 않았습니다.')
    st.download_button('성능 기록 JSON 저장',resource_bytes('assets/test_metrics.json'),'random_forest_test_metrics.json','application/json',key='download_metrics')


def clear_simulation():
    st.session_state.pop('simulation_result',None)
    st.session_state.pop('simulation_error',None)

def simulation_page():
    st.subheader('하이퍼파라미터에 따라 모델은 어떻게 달라질까요?')
    st.write('노트북에서 학습한 랜덤포레스트 8개 설정을 불러와 검증 성능과 새 입력의 점수를 비교합니다.')
    a,b,c=st.columns(3)
    n=a.selectbox('나무 수', [100,200],index=1,key='sim_trees',on_change=clear_simulation)
    d=b.selectbox('최대 깊이', [8,12],index=1,key='sim_depth',on_change=clear_simulation)
    l=c.selectbox('잎의 최소 표본 수', [5,20],key='sim_leaf',on_change=clear_simulation)
    candidate_id=f'rf_t{n}_d{d}_l{l}'
    record=variant_record(candidate_id)
    st.caption('공통 설정: class_weight=balanced, random_state=42. 주모델 설정은 노트북의 검증 결과로 확정합니다.')
    a,b,c=st.columns(3)
    a.metric('선택 설정의 검증 평균 F1',f"{record['macro_f1']:.3f}")
    b.metric('선택 설정의 실패 재현율',f"{record['recall_failure']:.1%}")
    c.metric('선택 설정의 실패 정밀도',f"{record['precision_failure']:.1%}")
    with st.container(border=True):
        st.write('같은 프로젝트 입력으로 비교하기 · 미국 / USD 기준')
        a,b,c=st.columns(3)
        category=a.selectbox('비교할 분야',list(CATEGORIES),format_func=lambda x:CATEGORIES[x],key='sim_category',on_change=clear_simulation)
        goal=b.number_input('비교 목표금액 (USD)',min_value=0.,value=5000.,step=100.,key='sim_goal',on_change=clear_simulation)
        days=c.number_input('비교 기간 (일)',min_value=1,max_value=60,value=30,key='sim_days',on_change=clear_simulation)
        if st.button('선택 설정으로 비교하기',key='simulate',type='primary'):
            clear_simulation()
            try:
                chosen=analyze(category,goal,days,'native',None,'US','USD',candidate_id=candidate_id)
                primary=analyze(category,goal,days,'native',None,'US','USD')
                st.session_state['simulation_result']={'chosen':chosen,'primary':primary}
            except ValueError as e:st.session_state['simulation_error']=str(e)
        if st.session_state.get('simulation_error'):st.error(st.session_state['simulation_error'])
        if st.session_state.get('simulation_result'):
            result=st.session_state['simulation_result'];a,b=st.columns(2)
            a.metric('선택 설정의 성공 점수',f"{result['chosen']['success_score']:.1%}")
            b.metric('최종 주모델의 성공 점수',f"{result['primary']['success_score']:.1%}")
            st.write('선택 설정의 판단: '+result['chosen']['predicted_label'])
            st.caption('실제로 저장된 서로 다른 모델을 같은 입력으로 비교한 결과입니다. 점수는 보정된 성공확률이 아닙니다.')
    tuning=resource_csv('assets/rf_tuning.csv')
    view=tuning[['candidate_id','n_estimators','max_depth','min_samples_leaf','macro_f1','recall_failure']].rename(columns={'candidate_id':'설정','n_estimators':'나무 수','max_depth':'최대 깊이','min_samples_leaf':'최소 잎 표본','macro_f1':'검증 평균 F1','recall_failure':'실패 재현율'})
    st.dataframe(view,hide_index=True,width='stretch',column_config={c:st.column_config.NumberColumn(format='%.3f') for c in ['검증 평균 F1','실패 재현율']})
    st.bar_chart(view.set_index('설정')[['검증 평균 F1','실패 재현율']], stack=False)

def learning_page():
    st.subheader('이미지의 10단계 순서로 다시 실행한 프로젝트')
    steps=['문제 정의','데이터 수집','EDA','데이터 분할','전처리 Pipeline','모델 선택','학습 Fit','평가 및 튜닝','모델 저장 Joblib','배포 Streamlit']
    st.write(' → '.join(f'{i}. {name}' for i,name in enumerate(steps,1)))
    st.dataframe(pd.DataFrame([
        ['crowdfunding_analysis.ipynb','EDA·분할·전처리·모델 비교·학습·평가·튜닝·저장·시각화'],
        ['streamlit_crowdfunding.py','모델 로드·프로토타이핑·저장된 설정 후보 시뮬레이션·시각화']
    ],columns=['파일','역할']),hide_index=True,width='stretch')
    lessons=pd.DataFrame([
        ['분류와 랜덤포레스트','성공=1/실패=0, RF 주모델','6·10~14·19쪽'],
        ['EDA와 전처리','분포·상관관계·로그변환·표준화·원-핫 인코딩','37~43쪽'],
        ['불균형 처리','class_weight=balanced','44쪽'],
        ['성능 평가','정확도·정밀도·재현율·F1','17·21·28~32쪽'],
        ['비교·튜닝·중요도·저장','검증자료로 RF 설정 선택, 시험 뒤 저장','45쪽']
    ],columns=['배운 내용','적용','교재'])
    st.dataframe(lessons,hide_index=True,width='stretch')
    spec=model_spec();p=spec['parameters']
    st.write(f"선택된 RF: 나무 {p['n_estimators']}개, 깊이 {p['max_depth']}, 최소 잎 표본 {p['min_samples_leaf']}. 검증 평균 F1 {spec['validation_macro_f1']:.4f}.")
    with st.expander('실제 학습 순서 기록'):
        log=resource_json('assets/workflow_execution.json')
        st.dataframe(pd.DataFrame(log['steps'])[['step','name','completed_utc','detail']],hide_index=True,width='stretch')
        st.caption('노트북의 10단계 기록은 배포 준비 확인입니다. 공개 배포는 별도의 실제 URL 확인으로 판정합니다.')
    with st.expander('기본모델 검증 성능 비교'):
        table=resource_csv('assets/model_comparison.csv')
        st.dataframe(table[['model','accuracy','precision_failure','recall_failure','macro_f1']],hide_index=True,width='stretch')
    with st.expander('변수 중요도'):
        st.image(resource_bytes('figures/F11_feature_importance.png'))
        st.caption('검증자료 고정 표본 3,000건·3회 반복. 변수와 예측의 관련성을 보여주며 인과효과는 아닙니다.')
    st.markdown('[Web Robots Kickstarter 자료](https://webrobots.io/kickstarter-datasets/) · 2026-10-12 스냅샷')
    st.write('기존 수집 자료를 재사용해 시간순 분할과 학습을 다시 실행했습니다. 학습·검증·시험 32,303건, 참고용 국내 사례 100건입니다.')
    st.caption('노트북 기본 실행은 실제 재학습입니다. 이 앱은 저장모델을 읽어서 예측합니다. 국내 성공확률 보정이나 PR/AP 분석은 사용하지 않습니다.')

def main():
    st.set_page_config(page_title='모금나침반 · 창작 펀딩 계획',page_icon='🧭',layout='wide',initial_sidebar_state='collapsed')
    st.markdown(CSS,unsafe_allow_html=True)
    st.markdown('<div class="hero"><div class="eyebrow">CREATIVE FUNDING · PLAN WITH EVIDENCE</div><h1>모금나침반</h1><p>창작을 시작하기 전, 목표와 기간을 함께 살펴보세요.<br>학습한 데이터로 계획을 비교하고, 실제 국내 사례를 확인합니다.</p><span class="tag">랜덤포레스트 · 창작 분야 13종</span></div>',unsafe_allow_html=True)
    try:load_model()
    except (OSError,ValueError,RuntimeError,KeyError) as e:
        st.error(f'예측 모델을 불러오지 못했습니다. {e}')
        st.info('제공한 앱 폴더 전체와 requirements.txt를 사용해 주세요.');st.stop()
    tabs=st.tabs(['계획 분석','모델 설정 비교','국내 사례','모델 성능','10단계 개발 과정'])
    with tabs[0]:plan_page()
    with tabs[1]:simulation_page()
    with tabs[2]:domestic_page()
    with tabs[3]:performance_page()
    with tabs[4]:learning_page()
    st.divider();st.caption('모금나침반 · 백가연 · 머신러닝 서비스모델 과제 | 해외 데이터 기반 교육용 서비스')

if __name__=='__main__':
    if '--check' in sys.argv:print(json.dumps(check_runtime(),ensure_ascii=False,indent=2))
    else:main()
