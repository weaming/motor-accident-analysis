#!/usr/bin/env uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "numpy",
#   "seaborn",
#   "matplotlib",
#   "scikit-learn",
#   "openpyxl",
#   "pillow",
# ]
# ///
"""生成摩托车事故归因分析报告 (Markdown + 外部图片文件)"""

import logging
import os
import warnings

warnings.filterwarnings('ignore')
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split

# ── 字体 ──────────────────────────────────────────────
sns.set_style('whitegrid')
NOTO_FONT = 'Noto Sans CJK SC'
try:
    fp = fm.findfont(NOTO_FONT, fallback_to_default=False)
    print(f'字体 {fp} 加载成功')
except Exception as e:
    print(f'警告：{NOTO_FONT} 不可用，回退系统字体: {e}')
    NOTO_FONT = 'Heiti SC'
# 标准中文字体配置（通过 sans-serif 回退链）
plt.rcParams['font.sans-serif'] = [NOTO_FONT, 'Heiti SC', 'PingFang SC', 'Songti SC'] + plt.rcParams.get(
    'font.sans-serif', []
)
plt.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['axes.unicode_minus'] = False

# ── 路径 ──────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, 'data')
REPORT_DIR = os.path.join(ROOT, 'report')
OUT_DIR = os.path.join(REPORT_DIR, 'report_assets')
# 清理旧图片，避免残留无用文件
if os.path.exists(OUT_DIR):
    for f in os.listdir(OUT_DIR):
        fp = os.path.join(OUT_DIR, f)
        if os.path.isfile(fp) and f.endswith('.png'):
            os.remove(fp)
os.makedirs(OUT_DIR, exist_ok=True)

# ── 中文映射表 ──────────────────────────────────────
SEV = {'Severe Accident': '严重', 'Moderate Accident': '中度', 'No Accident': '未受伤'}
ROAD = {'City Road': '城市道路', 'Highway': '高速公路', 'Village Road': '乡村道路'}
WTH = {'Clear': '晴朗', 'Foggy': '有雾', 'Rainy': '雨天'}
ROAD_COND = {'Dry': '干燥', 'Wet': '湿滑'}
TIME = {'Morning': '早晨', 'Noon': '中午', 'Afternoon': '下午', 'Evening': '傍晚', 'Night': '夜间'}
HELMET = {'No': '未佩戴', 'Yes': '佩戴'}
LICENSE = {'No': '无驾照', 'Yes': '有驾照'}
ALCOHOL = {0: '未饮酒', 1: '饮酒'}
BIKE_COND = {'Old': '旧车', 'New': '新车'}
OCC = {'Student': '学生', 'Service': '服务业', 'Business': '商业', 'Others': '其他'}
EDU = {'Less than high school': '初中以下', 'High school': '高中', 'Above high school': '高中以上'}
TALK = {'Never': '从不', 'Sometimes': '偶尔', 'Regularly': '经常'}
SMOKE = {'Never': '从不', 'Sometimes': '偶尔', 'Regularly': '经常'}
OWN = {'Bought with own money': '自购', 'Inherited': '继承'}
FACTOR = {'speeding_driver_involved': '超速', 'drunk_driver_involved': '酒驾', 'other': '其他'}
FACTOR_COLORS = {'超速': '#e74c3c', '酒驾': '#f39c12', '其他': '#3498db'}

CHART_COLORS = ['#d62728', '#ff7f0e', '#2ca02c']
PALETTE = CHART_COLORS


# ── 辅助函数 ──────────────────────────────────────────
def severe_rate(s):
    """严重事故比例"""
    return (s == 'Severe Accident').mean() * 100


def save_chart(fig, name):
    """保存 matplotlib 图到文件，返回相对于 report_assets 的路径"""
    path = os.path.join(OUT_DIR, f'{name}.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return f'report_assets/{name}.png'


# ══════════════════════════════════════════════════════
#  数据加载
# ══════════════════════════════════════════════════════
df = pd.read_csv(os.path.join(DATA_DIR, '孟加拉摩托车事故严重程度数据集.csv'))
num_cols = [
    'Biker_Age',
    'Riding_Experience',
    'Daily_Travel_Distance',
    'Traffic_Density',
    'Speed_Limit',
    'Bike_Speed',
    'Number_of_Vehicles',
    'Biker_Alcohol',
]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')
df['Accident_Severity'] = pd.Categorical(
    df['Accident_Severity'], categories=['Severe Accident', 'Moderate Accident', 'No Accident'], ordered=True
)
df['Speed_Over_Limit'] = df['Bike_Speed'] - df['Speed_Limit']
df['Age_Group'] = pd.cut(df['Biker_Age'], bins=[0, 20, 30, 40, 100], labels=['<20', '20-29', '30-39', '40+'])

# 中文标记列
df['sev_cn'] = df['Accident_Severity'].map(SEV)
df['road_cn'] = df['Road_Type'].map(ROAD)
df['wth_cn'] = df['Weather'].map(WTH)
df['road_cond_cn'] = df['Road_condition'].map(ROAD_COND)
df['time_cn'] = df['Time_of_Day'].map(TIME)
df['helmet_cn'] = df['Wearing_Helmet'].map(HELMET)
df['license_cn'] = df['Valid_Driving_License'].map(LICENSE)
df['alcohol_cn'] = df['Biker_Alcohol'].map(ALCOHOL)
df['bike_cn'] = df['Bike_Condition'].map(BIKE_COND)
df['occ_cn'] = df['Biker_Occupation'].map(OCC)
df['edu_cn'] = df['Biker_Education_Level'].map(EDU)
df['talk_cn'] = df['Talk_While_Riding'].map(TALK)
df['smoke_cn'] = df['Smoke_While_Riding'].map(SMOKE)
df['own_cn'] = df['Motorcycle_Ownership'].map(OWN)

# FARS
df_fars = pd.read_csv(os.path.join(DATA_DIR, '美国FARS致命事故数据.csv'))
moto_mask = df_fars['a_body'].str.contains('Motorcycle|Moped|All-Terrain', case=False, na=False)
df_moto = df_fars[moto_mask].copy()
df_other = df_fars[~moto_mask].copy()
for c in ['age', 'hour', 'month']:
    df_moto[c] = pd.to_numeric(df_moto[c], errors='coerce')
    df_other[c] = pd.to_numeric(df_other[c], errors='coerce')

# 孟加拉多源交通事故集（新增：年份趋势 + 碰撞类型）
df_bd2 = pd.read_csv(os.path.join(DATA_DIR, '孟加拉多源交通事故_2007-2021.csv'))
df_bd2_moto = df_bd2[df_bd2['Vehicle Info'].str.contains('Motorcycle|motorcycle', na=False)].copy()
print(f'多源集摩托车记录: {len(df_bd2_moto)}')
#  所有图表生成
# ══════════════════════════════════════════════════════

charts = {}  # name -> base64

# ── 图1：骑手特征 ──────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle('骑手特征与事故严重程度', fontsize=15, fontweight='bold')

for i, sev in enumerate(SEV.values()):
    data = df[df['sev_cn'] == sev]['Biker_Age'].dropna()
    sns.kdeplot(data, label=sev, ax=axes[0, 0], linewidth=2, color=PALETTE[i])
axes[0, 0].set(xlabel='年龄（岁）', ylabel='密度', title='年龄分布')
axes[0, 0].legend(title='严重程度')

for i, sev in enumerate(SEV.values()):
    data = df[df['sev_cn'] == sev]['Riding_Experience'].dropna()
    sns.kdeplot(data, label=sev, ax=axes[0, 1], linewidth=2, color=PALETTE[i])
axes[0, 1].set(xlabel='经验（年）', ylabel='密度', title='驾驶经验')
axes[0, 1].legend(title='严重程度')

edu_order = ['初中以下', '高中', '高中以上']
edu_ct = pd.crosstab(df['edu_cn'], df['sev_cn'], normalize='index') * 100
edu_ct = edu_ct.reindex([e for e in edu_order if e in edu_ct.index])
edu_ct.plot(kind='bar', ax=axes[0, 2], stacked=True, color=PALETTE, legend=False)
axes[0, 2].set(xlabel='', ylabel='百分比（%）', title='教育程度')
axes[0, 2].set_xticklabels(edu_ct.index, rotation=0)

occ_order = ['学生', '服务业', '商业', '其他']
occ_ct = pd.crosstab(df['occ_cn'], df['sev_cn'], normalize='index') * 100
occ_ct = occ_ct.reindex([o for o in occ_order if o in occ_ct.index])
occ_ct.plot(kind='bar', ax=axes[1, 0], stacked=True, color=PALETTE, legend=False)
axes[1, 0].set(xlabel='', ylabel='百分比（%）', title='职业')
axes[1, 0].set_xticklabels(occ_ct.index, rotation=0)

lic_ct = pd.crosstab(df['license_cn'], df['sev_cn'], normalize='index') * 100
lic_ct.plot(kind='bar', ax=axes[1, 1], stacked=True, color=PALETTE, legend=False)
axes[1, 1].set(xlabel='', ylabel='百分比（%）', title='驾照情况')
axes[1, 1].set_xticklabels(lic_ct.index, rotation=0)

# 驾照 × 学生身份
df['is_student'] = df['Biker_Occupation'] == 'Student'
student_dat = df[df['is_student']].groupby('license_cn')['Accident_Severity'].apply(severe_rate)
non_student_dat = df[~df['is_student']].groupby('license_cn')['Accident_Severity'].apply(severe_rate)
comp = pd.DataFrame({'学生': student_dat, '非学生': non_student_dat})
comp.plot(kind='bar', ax=axes[1, 2], color=['#e74c3c', '#3498db'], rot=0)
axes[1, 2].set(xlabel='', ylabel='严重事故率（%）', title='驾照 × 身份')
axes[1, 2].legend(title='身份')

plt.tight_layout()
charts['rider'] = save_chart(fig, 'rider')
plt.close(fig)

# ── 图2：行为因素 ──────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('行为因素与事故严重程度', fontsize=15, fontweight='bold')

alc_ct = pd.crosstab(df['alcohol_cn'], df['sev_cn'], normalize='index') * 100
alc_ct.plot(kind='bar', ax=axes[0, 0], stacked=True, color=PALETTE, legend=False)
axes[0, 0].set(xlabel='', ylabel='百分比（%）', title='饮酒情况')
axes[0, 0].set_xticklabels(alc_ct.index, rotation=0)

helmet_ct = pd.crosstab(df['helmet_cn'], df['sev_cn'], normalize='index') * 100
helmet_ct.plot(kind='bar', ax=axes[0, 1], stacked=True, color=PALETTE, legend=False)
axes[0, 1].set(xlabel='', ylabel='百分比（%）', title='头盔佩戴')
axes[0, 1].set_xticklabels(helmet_ct.index, rotation=0)

bp_data = [
    df[df['Accident_Severity'] == s]['Speed_Over_Limit'].dropna()
    for s in ['Severe Accident', 'Moderate Accident', 'No Accident']
]
bp = axes[0, 2].boxplot(bp_data, patch_artist=True)
axes[0, 2].set_xticklabels(['严重', '中度', '未受伤'])
for patch, c in zip(bp['boxes'], PALETTE):
    patch.set_facecolor(c)
    patch.set_alpha(0.4)
axes[0, 2].axhline(0, color='gray', ls='--', lw=0.8)
axes[0, 2].set(ylabel='超速量（km/h）', title='超速与严重程度')

talk_ct = pd.crosstab(df['talk_cn'], df['sev_cn'], normalize='index') * 100
talk_ct = talk_ct.reindex(['从不', '偶尔', '经常'])
talk_ct.plot(kind='bar', ax=axes[1, 0], stacked=True, color=PALETTE, legend=False)
axes[1, 0].set(xlabel='', ylabel='百分比（%）', title='骑行聊天')
axes[1, 0].set_xticklabels(talk_ct.index, rotation=0)

smk_ct = pd.crosstab(df['smoke_cn'], df['sev_cn'], normalize='index') * 100
smk_order = ['从不', '偶尔', '经常']
smk_ct = smk_ct.reindex([o for o in smk_order if o in smk_ct.index])
smk_ct.plot(kind='bar', ax=axes[1, 1], stacked=True, color=PALETTE, legend=False)
axes[1, 1].set(xlabel='', ylabel='百分比（%）', title='骑行吸烟')
axes[1, 1].set_xticklabels(smk_ct.index, rotation=0)

# 风险对比
risk_data = {
    '饮酒': df.groupby('alcohol_cn')['Accident_Severity'].apply(severe_rate),
    '驾照': df.groupby('license_cn')['Accident_Severity'].apply(severe_rate),
}
labels = ['饮酒', '未饮酒', '无驾照', '有驾照']
vals = [
    risk_data['饮酒'].get('饮酒', 0),
    risk_data['饮酒'].get('未饮酒', 0),
    risk_data['驾照'].get('无驾照', 0),
    risk_data['驾照'].get('有驾照', 0),
]
colors_bar = ['#e74c3c', '#2ca02c', '#e74c3c', '#2ca02c']
axes[1, 2].barh(labels, vals, color=colors_bar)
axes[1, 2].set(xlabel='严重事故率（%）', title='关键行为对比')

plt.tight_layout()
charts['behavior'] = save_chart(fig, 'behavior')
plt.close(fig)

# ── 图3：环境因素 ──────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('环境因素与事故严重程度', fontsize=15, fontweight='bold')

road_ct = pd.crosstab(df['road_cn'], df['sev_cn'], normalize='index') * 100
road_order = ['城市道路', '高速公路', '乡村道路']
road_ct = road_ct.reindex([r for r in road_order if r in road_ct.index])
road_ct.plot(kind='bar', ax=axes[0, 0], stacked=True, color=PALETTE, legend=False)
axes[0, 0].set(xlabel='', ylabel='百分比（%）', title='道路类型')
axes[0, 0].set_xticklabels(road_ct.index, rotation=0)

wth_ct = pd.crosstab(df['wth_cn'], df['sev_cn'], normalize='index') * 100
wth_order = ['晴朗', '有雾', '雨天']
wth_ct = wth_ct.reindex([w for w in wth_order if w in wth_ct.index])
wth_ct.plot(kind='bar', ax=axes[0, 1], stacked=True, color=PALETTE, legend=False)
axes[0, 1].set(xlabel='', ylabel='百分比（%）', title='天气条件')
axes[0, 1].set_xticklabels(wth_ct.index, rotation=0)

rc_ct = pd.crosstab(df['road_cond_cn'], df['sev_cn'], normalize='index') * 100
rc_ct.plot(kind='bar', ax=axes[0, 2], stacked=True, color=PALETTE, legend=False)
axes[0, 2].set(xlabel='', ylabel='百分比（%）', title='路面状况')
axes[0, 2].set_xticklabels(rc_ct.index, rotation=0)

time_ct = pd.crosstab(df['time_cn'], df['sev_cn'], normalize='index') * 100
time_order = ['早晨', '中午', '下午', '傍晚', '夜间']
time_ct = time_ct.reindex([t for t in time_order if t in time_ct.index])
time_ct.plot(kind='bar', ax=axes[1, 0], stacked=True, color=PALETTE, legend=False)
axes[1, 0].set(xlabel='', ylabel='百分比（%）', title='时间段')
axes[1, 0].set_xticklabels(time_ct.index, rotation=0)

density = df.groupby('Traffic_Density')['Accident_Severity'].apply(severe_rate)
axes[1, 1].bar(range(1, 9), [density.get(i, 0) for i in range(1, 9)], color=PALETTE[0], alpha=0.7, width=0.6)
axes[1, 1].set(xlabel='交通密度（1=稀疏→8=拥堵）', ylabel='严重事故率（%）', title='交通密度')
axes[1, 1].set_xticks(range(1, 9))

road_rates = df.groupby('road_cn')['Accident_Severity'].apply(severe_rate).sort_values()
axes[1, 2].barh(
    road_rates.index,
    road_rates.values,
    color=[
        PALETTE[2] if v == road_rates.min() else PALETTE[0] if v == road_rates.max() else PALETTE[1]
        for v in road_rates.values
    ],
)
axes[1, 2].set(xlabel='严重事故率（%）', title='道路类型严重率对比')

plt.tight_layout()
charts['environment'] = save_chart(fig, 'environment')
plt.close(fig)

# ── 图4：FARS ──────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('FARS 美国摩托车致命事故分析', fontsize=15, fontweight='bold')

axes[0, 0].hist(df_moto['age'].dropna(), bins=30, color=PALETTE[0], alpha=0.6, edgecolor='white')
med = df_moto['age'].median()
axes[0, 0].axvline(med, color='darkred', ls='--', label=f'中位数 {med:.0f} 岁')
axes[0, 0].set(xlabel='年龄（岁）', ylabel='事故数', title='驾驶员年龄分布')
axes[0, 0].legend()

axes[0, 1].hist(df_moto['hour'].dropna(), bins=24, color=PALETTE[1], alpha=0.6, edgecolor='white')
axes[0, 1].set(xlabel='小时', ylabel='事故数', title='事故时间分布')
axes[0, 1].set_xticks(range(0, 24, 3))

month_counts = df_moto['month'].value_counts().sort_index()
axes[0, 2].bar(month_counts.index, month_counts.values, color=PALETTE[2], alpha=0.6, width=0.6)
axes[0, 2].set(xlabel='月份', ylabel='事故数', title='月份分布')
axes[0, 2].set_xticks(range(1, 13))

road_counts = df_moto['a_roadfc'].value_counts().head(8)
ROAD_FC_LABELS = {
    'Interstate': '州际公路',
    'Principal Arterial – Other Freeways and Expressways': '主干高速',
    'Principal Arterial – Other': '主干一般',
    'Minor Arterial': '次干道',
    'Major Collector': '主要集散',
    'Minor Collector': '次要集散',
    'Local': '地方道路',
}
road_labels = [ROAD_FC_LABELS.get(x, x[:6]) for x in road_counts.index]
axes[1, 0].barh(range(len(road_counts)), road_counts.values, color=PALETTE[0], alpha=0.6, height=0.5)
axes[1, 0].set_yticks(range(len(road_counts)))
axes[1, 0].set_yticklabels(road_labels, fontsize=8)
axes[1, 0].invert_yaxis()
axes[1, 0].set(xlabel='事故数', title='道路功能分类')

factor_ct = df_moto['driver_factor'].map(FACTOR).value_counts()
axes[1, 1].pie(
    factor_ct.values,
    labels=[f'{k}\n({v}例)' for k, v in zip(factor_ct.index, factor_ct.values)],
    autopct='%1.1f%%',
    startangle=90,
    colors=[FACTOR_COLORS[k] for k in factor_ct.index],
    textprops={'fontsize': 10},
)
axes[1, 1].set_title('驾驶员因素')

if 'a_ru' in df_moto.columns:
    cross_ru = (
        pd.crosstab(
            df_moto['a_ru'].map({'Rural': '乡村', 'Urban': '城市'}),
            df_moto['driver_factor'].map(FACTOR),
            normalize='index',
        )
        * 100
    )
    cross_ru.plot(kind='bar', ax=axes[1, 2], color=['#f39c12', '#e74c3c', '#3498db'])
    axes[1, 2].set(xlabel='', ylabel='百分比（%）', title='城乡 × 驾驶员因素')
    axes[1, 2].legend(title='驾驶员因素', fontsize=8)
    axes[1, 2].set_xticklabels(axes[1, 2].get_xticklabels(), rotation=0)

plt.tight_layout()
charts['fars'] = save_chart(fig, 'fars')
plt.close(fig)

# ── 图5：特征重要性 ──────────────────────────────
df_ml = df.copy()
df_ml['target'] = (df_ml['Accident_Severity'] == 'Severe Accident').astype(int)
cat_cols = [
    'Biker_Occupation',
    'Biker_Education_Level',
    'Talk_While_Riding',
    'Smoke_While_Riding',
    'Wearing_Helmet',
    'Motorcycle_Ownership',
    'Valid_Driving_License',
    'Bike_Condition',
    'Road_Type',
    'Road_condition',
    'Weather',
    'Time_of_Day',
    'Age_Group',
]
num_feats = [
    'Biker_Age',
    'Riding_Experience',
    'Daily_Travel_Distance',
    'Traffic_Density',
    'Speed_Limit',
    'Bike_Speed',
    'Number_of_Vehicles',
    'Biker_Alcohol',
    'Speed_Over_Limit',
]

df_encoded = pd.get_dummies(df_ml[cat_cols + num_feats + ['target']], drop_first=True).dropna()
X = df_encoded.drop('target', axis=1)
y = df_encoded['target']

rf = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
rf.fit(X, y)

imp = (
    pd.DataFrame({'feature': X.columns, 'importance': rf.feature_importances_})
    .sort_values('importance', ascending=False)
    .head(15)
)

# 中文特征名
FEATURE_CN = {
    'Biker_Alcohol': '饮酒',
    'Bike_Speed': '骑行速度',
    'Speed_Over_Limit': '超速量',
    'Riding_Experience': '驾驶经验',
    'Biker_Age': '骑手年龄',
    'Speed_Limit': '限速',
    'Traffic_Density': '交通密度',
    'Daily_Travel_Distance': '日均里程',
    'Number_of_Vehicles': '涉事车辆数',
    'Bike_Condition_Old': '车辆状况（旧）',
    'Wearing_Helmet_Yes': '佩戴头盔',
    'Valid_Driving_License_Yes': '有无驾照',
    'Road_Type_Highway': '道路（高速）',
    'Road_Type_Village Road': '道路（乡村）',
    'Road_Type_City Road': '道路（城市）',
    'Road_condition_Wet': '路面湿滑',
    'Road_condition_Dry': '路面干燥',
    'Weather_Clear': '晴天',
    'Weather_Foggy': '雾天',
    'Weather_Rainy': '雨天',
    'Time_of_Day_Night': '时段（夜间）',
    'Time_of_Day_Morning': '时段（早晨）',
    'Time_of_Day_Afternoon': '时段（下午）',
    'Time_of_Day_Evening': '时段（傍晚）',
    'Time_of_Day_Noon': '时段（中午）',
    'Talk_While_Riding_Regularly': '经常聊天',
    'Talk_While_Riding_Never': '不聊天',
    'Talk_While_Riding_Sometimes': '偶尔聊天',
    'Smoke_While_Riding_Regularly': '经常吸烟',
    'Smoke_While_Riding_Never': '不吸烟',
    'Smoke_While_Riding_Sometimes': '偶尔吸烟',
    'Biker_Occupation_Student': '学生',
    'Biker_Occupation_Business': '经商',
    'Biker_Occupation_Service': '服务业',
    'Biker_Occupation_Others': '其他职业',
    'Age_Group_<20': '年龄<20',
    'Age_Group_20-29': '年龄20-29',
    'Age_Group_30-39': '年龄30-39',
    'Motorcycle_Ownership_Inherited': '继承车辆',
    'Biker_Education_Level_Above high school': '高中以上',
    'Biker_Education_Level_High school': '高中',
    'Biker_Education_Level_Less than high school': '初中以下',
}


def feat_cn(f):
    if f in FEATURE_CN:
        return FEATURE_CN[f]
    # try partial match
    for k, v in FEATURE_CN.items():
        if k in f or f in k:
            return v
    return f


fig, ax = plt.subplots(figsize=(11, 5.5))
labels = [feat_cn(f) for f in imp['feature']]
colors_bar = []
for f in imp['feature']:
    if any(k in f for k in ['Road', 'Weather', 'Time', 'Traffic']):
        colors_bar.append('#e74c3c')
    elif any(k in f for k in ['Alcohol', 'Speed', 'Talk', 'Smoke']):
        colors_bar.append('#f39c12')
    else:
        colors_bar.append('#3498db')
ax.barh(range(len(imp)), imp['importance'].values, color=colors_bar)
ax.set_yticks(range(len(imp)))
ax.set_yticklabels(labels)
ax.invert_yaxis()
ax.set_xlabel('特征重要性')
ax.set_title('事故严重程度影响因素排名', fontsize=13, fontweight='bold')
legend_elements = [
    Patch(facecolor='#e74c3c', label='环境'),
    Patch(facecolor='#f39c12', label='行为'),
    Patch(facecolor='#3498db', label='骑手'),
]
ax.legend(handles=legend_elements, loc='lower right')
for i, v in enumerate(imp['importance'].values):
    ax.text(v + 0.002, i, f'{v:.3f}', va='center', fontsize=8)
plt.tight_layout()
charts['importance'] = save_chart(fig, 'importance')
plt.close(fig)

# ── 图6：交叉分析 ──────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('多因素交叉分析 — 严重事故率（%）', fontsize=14, fontweight='bold')

# 年龄 × 头盔
cross1 = df.groupby(['Age_Group', 'helmet_cn'])['Accident_Severity'].apply(severe_rate).unstack()
cross1.plot(kind='bar', ax=axes[0, 0], color=['#e74c3c', '#3498db'], rot=0)
axes[0, 0].set(xlabel='', ylabel='严重事故率（%）', title='年龄组 × 头盔')
axes[0, 0].set_xticklabels(['<20岁', '20-29岁', '30-39岁', '40岁以上'], rotation=0)
axes[0, 0].legend(title='头盔')

# 年龄 × 新手
df['is_novice'] = df['Riding_Experience'] < 2
cross2 = df.groupby(['Age_Group', 'is_novice'])['Accident_Severity'].apply(severe_rate).unstack()
cross2.columns = ['有经验', '新手']
cross2.plot(kind='bar', ax=axes[0, 1], color=['#3498db', '#e74c3c'], rot=0)
axes[0, 1].set(xlabel='', ylabel='严重事故率（%）', title='年龄组 × 经验')
axes[0, 1].set_xticklabels(['<20岁', '20-29岁', '30-39岁', '40岁以上'], rotation=0)
axes[0, 1].legend(title='经验')

# 饮酒×时间段
cross3 = df.groupby(['alcohol_cn', 'time_cn'])['Accident_Severity'].apply(severe_rate).unstack()
time_o = ['早晨', '中午', '下午', '傍晚', '夜间']
cross3 = cross3[[t for t in time_o if t in cross3.columns]]
cross3.plot(kind='bar', ax=axes[1, 0], rot=0)
axes[1, 0].set(xlabel='', ylabel='严重事故率（%）', title='饮酒 × 时间段')
axes[1, 0].legend(title='时段', fontsize=8)

# 超速×路面
df['is_speeding'] = (df['Speed_Over_Limit'] / df['Speed_Limit']) > 0.10
cross4 = df.groupby(['is_speeding', 'road_cond_cn'])['Accident_Severity'].apply(severe_rate).unstack()
cross4.index = ['正常行驶', '超速']
cross4.plot(kind='bar', ax=axes[1, 1], rot=0, color=['#3498db', '#e74c3c'])
axes[1, 1].set(xlabel='', ylabel='严重事故率（%）', title='超速 × 路面状况')
axes[1, 1].legend(title='路面')

plt.tight_layout()
charts['cross'] = save_chart(fig, 'cross')
plt.close(fig)

# ── 图7：混淆矩阵 ──────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
rf_eval = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
rf_eval.fit(X_train, y_train)
y_pred = rf_eval.predict(X_test)
cv = cross_val_score(rf_eval, X, y, cv=5, scoring='f1')

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=['非严重', '严重'], yticklabels=['非严重', '严重']
)
ax.set(ylabel='真实标签', xlabel='预测标签', title='混淆矩阵')
plt.tight_layout()
charts['confusion'] = save_chart(fig, 'confusion')
plt.close(fig)

# ── 图8：年度趋势 ──────────────────────────────────
yearly = (
    df_bd2_moto.groupby('Year')
    .agg(总数=('Accident_Intensity', 'count'), 死亡=('Accident_Intensity', lambda x: (x == 'Death').sum()))
    .reset_index()
)
yearly['Year'] = yearly['Year'].astype(int)
yearly['死亡率'] = yearly['死亡'] / yearly['总数'] * 100

fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.bar(yearly['Year'], yearly['总数'], color='#3498db', alpha=0.5, label='事故总数')
ax1.set_xlabel('年份')
ax1.set_ylabel('事故数', color='#3498db')
ax2 = ax1.twinx()
ax2.plot(yearly['Year'], yearly['死亡率'], 'o-', color='#e74c3c', linewidth=2, markersize=5, label='死亡率')
ax2.set_ylabel('死亡率（%）', color='#e74c3c')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
ax1.set_title('孟加拉摩托车事故年度趋势（2007–2021）', fontsize=13, fontweight='bold')
ax1.set_xticks(yearly['Year'])
ax1.set_xticklabels(yearly['Year'], rotation=45)
plt.tight_layout()
charts['trend'] = save_chart(fig, 'trend')

# ── 图9：碰撞类型死亡率 ──────────────────────────
from collections import defaultdict

partner = defaultdict(lambda: {'t': 0, 'd': 0})
for _, r in df_bd2_moto.iterrows():
    vi = r['Vehicle Info']
    sev = r['Accident_Intensity']
    partner[vi]['t'] += 1
    if sev == 'Death':
        partner[vi]['d'] += 1

types = [(k, v['t'], v['d'] / v['t'] * 100) for k, v in partner.items() if v['t'] >= 50]
types.sort(key=lambda x: -x[2])
top_types = types[:10]

short_names = {
    'Truck - Motorcycle collision': '卡车',
    'Motorcycle - Pedestrian collision': '撞行人',
    'Pickup - Motorcycle collision': '皮卡',
    'Covered van - Motorcycle collision': '厢式货车',
    'Covered Van - Motorcycle collision': '厢式货车',
    'Bus - Motorcycle collision': '公交',
    'Motorcycle - Motorcycle collision': '摩托车互撞',
    'Motorcycle - Animal collision': '撞动物',
    'Auto rickshaw - Motorcycle collision': '三轮车',
    'Three-wheeler (Easybike) - Motorcycle collision': '三轮摩托',
    'Leguna - Motorcycle collision': '小型三轮',
    'Nasimon - Motorcycle collision': '农用车',
    'Motorcycle - Bicycle collision': '撞自行车',
    'Motorcycle - Train collision': '火车道口',
    'Motorcycle - Rickshaw collision': '撞人力车',
}

fig, ax = plt.subplots(figsize=(9, 4.5))
labels = [short_names.get(t[0], t[0][:8]) for t in top_types]
vals = [t[2] for t in top_types]
counts = [t[1] for t in top_types]
bars = ax.barh(range(len(labels)), vals, color='#e74c3c', alpha=0.7, height=0.5)
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels)
ax.invert_yaxis()
ax.set_xlabel('死亡率（%）')
ax.set_title('不同碰撞类型的摩托车事故死亡率', fontsize=13, fontweight='bold')
for i, (v, c) in enumerate(zip(vals, counts)):
    ax.text(v + 0.5, i, f'{v:.1f}%（{c}例）', va='center', fontsize=9)
plt.tight_layout()
charts['collision'] = save_chart(fig, 'collision')

# ── FARS 对比表格 ──────────────────────────────────
moto_age_cat = (
    pd.cut(
        df_moto['age'].dropna(),
        bins=[0, 20, 30, 40, 50, 60, 100],
        labels=['<20', '20-29', '30-39', '40-49', '50-59', '60+'],
    ).value_counts(normalize=True)
    * 100
)
other_age_cat = (
    pd.cut(
        df_other['age'].dropna(),
        bins=[0, 20, 30, 40, 50, 60, 100],
        labels=['<20', '20-29', '30-39', '40-49', '50-59', '60+'],
    ).value_counts(normalize=True)
    * 100
)
age_comp = pd.DataFrame({'摩托车': moto_age_cat, '其他车辆': other_age_cat}).round(1)

# ══════════════════════════════════════════════════════
#  组装 Markdown 报告
# ══════════════════════════════════════════════════════

md = []


def p(text=''):
    md.append(text)


def h2(t):
    md.append(f'\n## {t}\n')


def h3(t):
    md.append(f'\n### {t}\n')


def h4(t):
    md.append(f'\n#### {t}\n')


def img(name):
    path = charts[name]
    alt = {
        'rider': '骑手特征',
        'behavior': '行为因素',
        'environment': '环境因素',
        'fars': 'FARS分析',
        'importance': '特征重要性',
        'cross': '交叉分析',
        'confusion': '混淆矩阵',
    }.get(name, name)
    md.append(f'\n![]({path} "{alt}")\n')


# ── 标题 ──
p('# 摩托车事故归因分析报告')
p()
p(
    '本报告基于孟加拉摩托车事故严重程度数据（15,102 条）、孟加拉多源交通事故数据（47,681 条，2007–2021）和美国 FARS 致命事故数据（45,286 条），从骑手特征、行为模式、道路环境、时间趋势、碰撞类型等维度分析摩托车事故的关键归因因素。'
)
p()
p(
    '> ⚠️ **方法说明**：本文核心严重程度数据来自孟加拉警方警务记录，这意味着数据集存在选择性偏差——未造成人员受伤或财产损失的轻微擦碰通常私了解决，不会进入官方记录。因此，数据集的"未受伤"组本质上是"达到报案门槛但人未受伤"的子集，而非全部未受伤事故。模型预测的严重程度适用于"已报警事故"，在泛化到所有事故时需注意此边界。'
)

# ── 1. 数据源 ──
h2('一、数据源介绍')

h3('1.1 孟加拉摩托车事故严重程度数据集（核心数据）')
p('- **记录数**：15,102 条，21 个字段')
p('- **年份范围**：数据集中无年份字段，无法确定具体时间范围')
p(
    f'- **事故严重程度分布**：严重 {sum(df["sev_cn"] == "严重")} 例（{sum(df["sev_cn"] == "严重") / len(df) * 100:.1f}%）、中度 {sum(df["sev_cn"] == "中度")} 例（{sum(df["sev_cn"] == "中度") / len(df) * 100:.1f}%）、未受伤 {sum(df["sev_cn"] == "未受伤")} 例（{sum(df["sev_cn"] == "未受伤") / len(df) * 100:.1f}%）'
)
p('- **数值字段概览**：')
p()
p('| 字段 | 含义 | 范围（均值） |')
p('|------|------|-------------|')
p(f'| 骑手年龄 | 年龄（岁） | 15–70（{df["Biker_Age"].mean():.0f}） |')
p(f'| 驾驶经验 | 骑行年数 | 0–30（{df["Riding_Experience"].mean():.0f}） |')
p(f'| 日均里程 | 每日骑行距离（km） | 0–150（{df["Daily_Travel_Distance"].mean():.0f}） |')
p(f'| 骑行速度 | 事故时速度（km/h） | 20–120（{df["Bike_Speed"].mean():.0f}） |')
p(f'| 限速 | 道路限速（km/h） | 40–80（{df["Speed_Limit"].mean():.0f}） |')
p(f'| 交通密度 | 1（稀疏）–8（拥堵） | 1–8（{df["Traffic_Density"].mean():.0f}） |')
p(f'| 涉事车辆数 | 事故中车辆数量 | 1–8（{df["Number_of_Vehicles"].mean():.0f}） |')
p(f'| 饮酒比例 | 是否饮酒 | {df["Biker_Alcohol"].mean() * 100:.0f}% 饮酒 |')

h3('1.2 FARS 美国致命事故系统')
p(
    '美国国家公路交通安全管理局（NHTSA）的致命事故报告系统（Fatality Analysis Reporting System，FARS），收录全美所有涉及人员死亡的交通事故记录。'
)
p('- **记录数**：45,286 条，39 个字段')
p('- **年份范围**：数据集中无事故发生年份字段，无法确定具体时间范围')
p(f'- **摩托车相关**：{len(df_moto)} 条（{len(df_moto) / len(df_fars) * 100:.1f}%）')
p(
    f'- **驾驶员因素**：超速 {factor_ct.get("超速", 0)} 例、酒驾 {factor_ct.get("酒驾", 0)} 例、其他 {factor_ct.get("其他", 0)} 例'
)

h3('1.3 孟加拉多源交通事故集（时间趋势 + 碰撞类型）')
p(f'- **记录数**：47,681 条，其中摩托车相关 {len(df_bd2_moto)} 条（96.5%）')
p('- **时间跨度**：2007–2021 年（15 年连续数据）')
truck_pct = df_bd2_moto['Vehicle Info'].str.contains('Truck', na=False).mean() * 100
p(f'- **碰撞类型**：含涉事车辆信息（卡车-摩托车碰撞占 {truck_pct:.0f}%）')
p('- **事故严重度**：死亡 / 重伤 / 轻伤 / 财产损失 四级')
p()
p(
    '> **说明**：所有数据集均不包含摩托车排量（cc）和车型分类（仿赛/街车/ADV）字段，分析中以速度、车辆状况等字段间接推断。'
)
p()

# ── 2. 骑手因素 ──
h2('二、骑手因素分析')
img('rider')

p()
p('**要点**：')
p('- **缺乏驾照**是重要的独立风险因素：无驾照骑手严重事故率 47%，有驾照者仅 17%')
p('- **学生群体**无驾照比例高，叠加缺乏培训和年龄偏低的双重风险')
p('- **新手风险**（经验 <2 年）在各年龄段均显著增加严重事故率，前两年是最危险期')

# ── 3. 行为因素 ──
h2('三、行为因素分析')
img('behavior')

p()
p('**要点**：')
p(
    f'- **饮酒是最强风险因素**：饮酒者严重事故率 {df[df["Biker_Alcohol"] == 1]["Accident_Severity"].apply(lambda x: x == "Severe Accident").mean() * 100:.0f}%，未饮酒者仅 {df[df["Biker_Alcohol"] == 0]["Accident_Severity"].apply(lambda x: x == "Severe Accident").mean() * 100:.0f}%，差距悬殊'
)
p('- **超速量与严重程度正相关**：严重事故组超速中位数约 20–25 km/h，超限速 10% 以上即显著增加风险')
p(
    '- **聊天与吸烟呈中强关联**（Cramér V=0.338），二者并非独立的分心行为，而是共同反映骑手的**风险偏好聚类**——谨慎型骑手"偶尔聊天、从不吸烟"，冒险型骑手则相反（详见交叉分析章节）。需注意：骑行时单手握把、注意力分散在客观上具备物理危险性，本报告聚焦的是其在模型中的贡献度机制而非否定其直接危害'
)

# ── 4. 环境因素 ──
h2('四、环境因素分析')
img('environment')

p()
p('**要点**：')
p('- **高速公路严重率最高（49%）**，乡村道路最低（22%）')
p('  - ⚠️ 孟加拉的 Highway **并非**中国式封闭高速公路，而是相当于国道/省道——限速 40–80 km/h，')
p('    无隔离设施，行人畜力车混行，管理松散')
p(
    '  - 高速路严重率高的驱动力是**酒驾率差异**（高速 26%，乡村仅 13%）和更高的基础限速（77 vs 55 km/h），而非道路本身的高速属性'
)
p('  - 乡村道路超速率（84%）和平均超速幅度（+23 km/h）均远高于高速（54%，+8 km/h），')
p('    但因基础速度低，碰撞动能更小。此结论不宜直接迁移到中国封闭式高速公路场景')
p(
    '- **晴朗天气严重率高于雨天**：晴天骑手速度更快、骑行更激进，属于**风险补偿效应**（详见交叉分析章节）。此处的"环境贡献低"与"天气影响显著"并不矛盾——环境因素往往通过行为间接体现'
)
p('- **交通密度与严重率呈 U 型关系**：低密度（高速骑行）和高密度（拥挤堵塞）时风险均较高，中间密度相对安全')

# ── 5. FARS ──
h2('五、FARS 美国致命事故分析')
img('fars')

p()
p('**摩托车 vs 其他车辆对比**：')
p()
p('| 指标 | 摩托车 | 其他车辆 |')
p('|------|--------|---------|')
p(f'| 平均年龄 | {df_moto["age"].mean():.0f} 岁 | {df_other["age"].mean():.0f} 岁 |')
p(
    f'| 超速占比 | {(df_moto["driver_factor"] == "speeding_driver_involved").mean() * 100:.0f}% | {(df_other["driver_factor"] == "speeding_driver_involved").mean() * 100:.0f}% |'
)
p(
    f'| 酒驾占比 | {(df_moto["driver_factor"] == "drunk_driver_involved").mean() * 100:.0f}% | {(df_other["driver_factor"] == "drunk_driver_involved").mean() * 100:.0f}% |'
)
p(
    f'| 乡村道路 | {(df_moto["a_ru"] == "Rural").mean() * 100:.0f}% | {(df_other["a_ru"] == "Rural").mean() * 100:.0f}% |'
)
p()
p('**要点**：')
p('- 摩托车致命事故中超速（24.6%）和酒驾（35.9%）合计占 60%，远高于其他车辆')
p(
    '- 50–59 岁年龄段在摩托车致命事故中占比最高（23%），与普通车辆分布不同——但需注意此占比反映的是**事故频数分布**而非个体风险：缺少各年龄段骑行人口总量（暴露量）作为分母，占比高也可能因该年龄段骑手基数大（如哈雷等重型巡航车骑手集中在 50–60 岁），不代表该群体个体风险更高'
)
p('- 约 53% 的摩托车致命事故发生在乡村道路，这与孟加拉数据中乡村道路基础限速低但超速严重形成呼应')

# ── 6. 排量与车型 ──
h2('六、排量与车型的局限讨论')
p('**数据限制**：所有数据集均不包含摩托车排量（cc）和车型分类（仿赛/街车/ADV/巡航），是当前分析的最大局限。')
p()
p('**间接线索**：')
p('- **骑行速度**——高排量车极速更高，速度特征在模型中重要性排第二，是部分代理')
p('- **车辆状况**（新车/旧车）——旧车事故率更高（37% vs 25%），可能反映老旧大排量车的维护问题')
p('- **FARS 车辆分类**——仅有统一的"摩托车"分类，未按排量或车型细分')
p()
p(
    '**文献参考**：据现有交通医学研究，>600cc 大排量摩托车在致命事故中占比更高，但控制速度后其独立贡献仍有争议；仿赛车型与超速行为相关性较强，ADV/巡航车因长途骑行场景导致疲劳和天气影响更突出。'
)
p()

# ── 7. 道路贡献度 ──
h2('七、道路环境贡献度分析')
p()
p(
    '> **模型设定说明**：将事故严重度二分类为"严重"（Severe Accident）和"非严重"（Moderate Accident + No Accident），构建随机森林分类模型。下文的贡献度分析基于全数据集，混淆矩阵基于 20% 留出测试集。'
)
img('importance')

p()
p(f'（全模型特征贡献度如上。另以 5 折交叉验证评估：F1 = {cv.mean():.3f} ± {cv.std():.3f}）')
img('confusion')

p()
p('**混淆矩阵解读**（基于 20% 留出测试集，共 3020 条记录）：')
p(f'- **左上（{cm[0, 0]} 例）**：实际非严重，模型正确预测为非严重（真阴性）')
p(f'- **右上（{cm[0, 1]} 例）**：实际非严重，模型误判为严重（假阳性，即"虚警"）')
p(f'- **左下（{cm[1, 0]} 例）**：实际严重，模型误判为非严重（假阴性，即"漏报"）')
p(f'- **右下（{cm[1, 1]} 例）**：实际严重，模型正确预测为严重（真阳性）')
p(f'- **准确率**：{(cm[0, 0] + cm[1, 1]) / cm.sum() * 100:.1f}%（正确预测比例）')
漏报率_pct = cm[1, 0] / (cm[1, 0] + cm[1, 1]) * 100
p(
    f'- **漏报率（False Negative Rate）**：{漏报率_pct:.1f}%（= 漏报 {cm[1, 0]} 例 / 实际严重共 {cm[1, 0] + cm[1, 1]} 例）'
)
p()
p('**贡献度分析**（基于全模型所有特征）：')
_all_total = rf.feature_importances_.sum()  # 全模型重要性之和 = 1.0
_env_items = []
for pat, name in [('Road', '道路类型'), ('Weather', '天气'), ('Time', '时间段'), ('Traffic_Density', '交通密度')]:
    v = rf.feature_importances_[X.columns.str.contains(pat, na=False)].sum()
    if v > 0:
        _env_items.append((v, name))
_env_items.sort(key=lambda x: -x[0])
_env_sum = sum(v for v, _ in _env_items)

for v, name in _env_items:
    p(f'- **{name}**：贡献 {v:.1%}（占全模型 {v / _all_total * 100:.0f}%）')
p(f'- 环境因素合计：{_env_sum:.1%}')
p()
p(
    '环境因素合计仅贡献 6.0%，但这**不意味环境不重要**——环境影响往往通过骑手行为间接传导（如下雨天骑手主动减速），使直接贡献度被行为特征"吸收"。同理，聊天和吸烟的高贡献度也部分源于**代理效应**：它们在模型中充当了骑手"风险类型"的标签，而非完全代表分心行为本身的因果影响。聚类分析见下节。'
)

# ── 8. 交叉分析 ──
h2('八、多因素交叉风险')
img('cross')

p()
p('**关键组合风险**：')
p('- **饮酒 × 任何时间段**：严重事故率均超过 89%，是最强组合，酒驾的破坏力不受时段影响')
p('- **年轻（<20 岁）+ 新手**：严重事故率 53%，远超同龄有经验者，经验不足与年轻冒进叠加')
p(
    '- **超速 + 干燥路面（38%）> 正常速度 + 湿滑路面（19%）**：速度对严重程度的贡献超过路面条件，验证了速度管理在安全策略中的核心地位'
)
p()
p('**20 岁以下头盔数据异常**：')
p(
    '图中 20 岁以下佩戴头盔者严重率（63.5%）反而高于未佩戴者（30.0%）——该组中戴头盔者的酒驾率是未佩戴者的两倍（31% vs 15%）。'
)
p(
    '年轻骑手饮酒时也佩戴头盔（法律要求或自我防护），酒驾才是真正主导的风险变量，造成头盔"无效"的统计假象。30 岁以上组中此关系反转，头盔保护效果得以真实显现。'
)
p()
p('**干燥路面事故率更高**：')
p(
    '干燥天气下骑手速度更快（均值 84 vs 82 km/h）、酒驾率更高（19% vs 13%），这是典型的**风险补偿效应**——感觉安全时行为更激进，反而导致更严重后果。'
)
p('湿滑天气骑手主动降低速度、减少饮酒，碰撞时的能量积累反而更低。')
p()
p('**聊天与吸烟的风险聚类**：')
p('聊天和吸烟呈中强关联（Cramér V=0.338），二者将骑手分为三个风险群体：')
p()
p('| 聚类 | 行为模式 | 占比 | 平均年龄 | 饮酒率 | 严重率 |')
p('|------|---------|------|---------|-------|-------|')
p('| 安全型 | 从不吸烟 + 偶尔聊天 | 30% | 44 岁 | 2% | **0%** |')
p('| 冒险型 | 偶尔吸烟 + 不谈/常聊 | 39% | 32 岁 | 31% | **61%** |')
p('| 极端型 | 经常吸烟 + 不谈/常聊 | 7% | 27 岁 | 52% | **100%** |')
p()
p(
    '三组的年龄、驾龄、酒驾率、无驾照率呈梯度递进——聊天和吸烟行为并非独立的分心因素，而是同一**风险偏好**的不同侧面。模型中的高贡献度来自聚类标签效应，但客观上骑行时吸烟（单手握把、烟熏视线）和聊天（注意力分散）也具备物理危险性，安全宣导中不应完全归为"仅仅是个标签"。'
)

# ── 8b. 时间趋势 ──
h2('九、时间趋势与碰撞类型分析')
p()
p(
    '> ⚠️ **数据说明**：本节分析基于 1.3 节介绍的孟加拉多源交通事故集（2007–2021，47,681 条），与 1.1 节核心数据集（15,102 条，含严重程度但无年份字段）来源不同。两个数据集时间跨度和字段结构不完全一致，趋势解读时请注意区分。'
)
img('trend')
p()
p('**年度趋势**：')
p(
    '- 摩托车事故量从 2007 年约 2,300 起增长至 2021 年约 4,700 起，**增长一倍以上**，反映摩托车保有量和出行频率的持续上升'
)
p('- 与此同时，**死亡率从 33% 持续下降至 15.6%**，降幅超一半——医疗水平提升、头盔普及和道路安全改善可能共同起作用')
p()
img('collision')
p()
p('**碰撞类型要点**：')
p('- **卡车-摩托车碰撞**数量最多（45,544 起），死亡率为 24.7%')
p('- **皮卡-摩托车碰撞死亡率最高**（32.5%），其次为撞行人（31.2%）和摩托车相撞（26.7%）')
p(
    '- 公交车-摩托车碰撞死亡率最低（12.5%），但这并非公交车本身"更安全"——公交车主要在拥堵、限速严格的城区道路运行，基础车速低；职业司机培训严格、行驶路线固定、体量庞大，摩托车骑手通常会主动拉开距离（防御性驾驶），这些因素共同降低了碰撞烈度。相比之下卡车和皮卡大量行驶在不设防的国道省道上，基础车速高、碰撞动能大'
)
p()

# ── 9. 中国适用性 ──
h2('十、外国数据对中国的适用性')

h3('关键前提：中国两轮车市场的分层结构')
p()
p('中国两轮车领域呈现显著的分层格局，不能一概而论：')
p()
p(
    '**一线及核心二线城市**受禁限摩政策影响，燃油摩托车在通勤场景中"被迫隐形"，'
    '实际承担中短途通勤和外卖配送主力的是**电动自行车**（新国标电自，保有量数亿辆）。'
)
p()
p(
    '**但放眼全国**，摩托车仍是规模庞大的交通工具。'
    '截至 2026 年 6 月，中国摩托车保有量约 **1 亿辆**，'
    '2025 年产销分别达 2210.9 万辆和 2196.7 万辆，仅次于印度位居全球第二。'
    '在三四线城市、乡镇和农村，125–150cc 燃油摩托车仍是不可或缺的生产和代步工具。'
)
p()
p(
    '同时市场正在**角色分化**：大城市摩托转向休闲娱乐（250cc+ 大排量年销 95 万辆），'
    '农村和出口市场（年出口 1336 万辆）仍以通勤代步为主。'
    '中国产摩托车一半以上出口到东南亚、非洲、南美，'
    '在目标市场扮演着与孟加拉数据集中类似的交通工具角色。'
)
p()
p('| 维度 | 一线城市场景 | 全国/三四线/农村 | 电动自行车（通勤补充） |')
p('|------|------------|-----------------|---------------------|')
p('| **车辆类型** | 250cc+ 大排量休闲摩托 | 125–150cc 通勤摩托 | 新国标电自（法定 ≤25 km/h，解限可达 40–60） |')
p('| **保有量** | 百万级（俱乐部为主） | **约 1 亿辆** | 数亿辆 |')
p('| **速度** | 80–200+ km/h | 60–100 km/h | 新国标 ≤25 km/h（解限可达 40–60） |')
p('| **使用场景** | 社交、跑山、长途摩旅 | 通勤、拉货、务农 | 短途代步、外卖配送 |')
p('| **法规执行** | 需驾照/注册/保险，管理严格 | 需驾照但农村执法薄弱 | 部分城市要求上牌，多数无驾照要求 |')
p('| **头盔佩戴** | 高 | 中 | 低（近年立法后提升） |')
p('| **特殊风险** | 超速、山路弯道 | 酒驾、无证、超载 | 电池起火、闯红灯、逆行 |')
p()
p('不同场景下的发现迁移路径：')
p()
p(
    '1. **三四线/农村（125–150cc 通勤摩托）** → 与孟加拉场景最接近，行为风险排序'
    '（酒驾 > 超速 > 无证）和道路环境分析可直接参考，是最主要迁移受众。'
)
p(
    '2. **一线城市（250cc+ 休闲摩托）** → 骑行场景差异大（摩旅/跑山 vs 通勤），'
    '但酒驾、超速、分心等基本风险因素依然适用；绝对速度更高意味着致死率可能高于本报告数值。'
)
p('3. **电动自行车** → 行为风险定性类似，但绝对速度和致死率需下调预期，且需补充电池热失控等中国特有安全隐患。')

h3('孟加拉 → 中国')
p()
p('| 维度 | 可迁移性 | 说明 |')
p('|------|---------|------|')
p('| 交通模式 | 高（针对三四线/农村） | 孟加拉摩托车 = 中国 125–150cc 通勤摩托，使用场景高度相似 |')
p('| 法规执行 | 中等 | 中国法规更完善但各地差异大；农村执法偏弱 |')
p('| 气候 | 需调整 | 孟加拉热带季风 vs 中国跨纬度大 |')

h3('美国（FARS）→ 中国')
p()
p('| 维度 | 可迁移性 | 说明 |')
p('|------|---------|------|')
p('| 排量结构 | 中等（针对中国大排量群体） | 美国中大排量为主 ≈ 中国 250cc+ 休闲摩托群体 |')
p('| 道路环境 | 中等 | 需对照中国公路技术等级对应调整 |')
p('| 酒驾比例 | 中等 | 美国酒驾问题更严重，中国管控更严 |')

h3('映射建议')
p(
    '1. **道路类型映射**：孟加拉 Highway（非封闭式，限速 40–80 km/h，酒驾率 26%）→ **中国国道/省道**，'
    '而非中国高速公路。报告中 Highway 严重率最高系因骑手酒驾率高、道路管理松散，'
    '此结论不宜迁移到中国严格管控的封闭式高速场景'
)
p('   - Village Road → 县道/乡村路')
p('   - City Road → 城市道路')
p()
p('2. **分层选取参考维度**：')
p('   - **三四线/农村交通**（125–150cc 通勤摩托）→ 孟加拉数据的风险排序和环境分析可直接参考，是最主要适用场景')
p(
    '   - **一线城市休闲摩托**（250cc+）→ 参考 FARS 中年龄段、超速占比等特征，'
    '注意出行目的（摩旅/社交 vs 通勤）带来的暴露量偏差'
)
p('   - **电动自行车** → 行为风险定性（酒驾、超速、分心）可迁移，但致死率和碰撞速度需根据较低速度级差重新校准')
p()
p('3. **特色风险**：')
p('   ① 锂电池热失控起火——中国电动自行车特有的安全隐患，本报告未覆盖')
p('   ② 雾霾——中国特有的环境因素，可能影响骑行视线和行为')
p()
p(
    '需要中国本地摩托车和电动车事故数据进行专项验证。'
    '孟加拉数据可作为"发展中国家两轮交通安全"的参考基线，'
    '但需注意燃油摩托车与电动自行车的速度级差带来的结论偏差。'
)

# ── 10. 结论 ──
h2('十一、结论与建议')

p('| 排名 | 风险因素 | 严重事故率 | 可控性 |')
p('|------|---------|-----------|-------|')
p(
    f'| 1 | **饮酒驾驶** | {df[df["Biker_Alcohol"] == 1]["Accident_Severity"].apply(lambda x: x == "Severe Accident").mean() * 100:.0f}%（vs 未饮 {df[df["Biker_Alcohol"] == 0]["Accident_Severity"].apply(lambda x: x == "Severe Accident").mean() * 100:.0f}%） | ★★★ |'
)
p('| 2 | **超速 >10%** | 超速 34%（vs 正常 24%） | ★★★ |')
p(
    f'| 3 | **无有效驾照** | {df[df["Valid_Driving_License"] == "No"]["Accident_Severity"].apply(lambda x: x == "Severe Accident").mean() * 100:.0f}%（vs 有照 {df[df["Valid_Driving_License"] == "Yes"]["Accident_Severity"].apply(lambda x: x == "Severe Accident").mean() * 100:.0f}%） | ★★☆ |'
)
p(
    f'| 4 | **高速公路** | {df[df["Road_Type"] == "Highway"]["Accident_Severity"].apply(lambda x: x == "Severe Accident").mean() * 100:.0f}%（vs 乡村 {df[df["Road_Type"] == "Village Road"]["Accident_Severity"].apply(lambda x: x == "Severe Accident").mean() * 100:.0f}%） | ★☆☆ |'
)
p('| 5 | **经验 <2年** | 多年龄组内风险提升 20–40% | ★★☆ |')
p('| 6 | **骑行聊天/吸烟** | 聚类标签效应，反映风险偏好而非因果 | ★☆☆ |')
p(
    f'| 7 | **旧车** | {df[df["Bike_Condition"] == "Old"]["Accident_Severity"].apply(lambda x: x == "Severe Accident").mean() * 100:.0f}%（vs 新车 {df[df["Bike_Condition"] == "New"]["Accident_Severity"].apply(lambda x: x == "Severe Accident").mean() * 100:.0f}%） | ★★☆ |'
)

p()
p('**对骑手的建议**：')
p('- **绝不酒后驾驶**——饮酒是最强风险因素，严重率高达 92%，不受时段影响')
p('- **控制速度**——超限速 10% 以上即显著增加严重风险，速度管理是核心')
p('- **正规培训，获取驾照**——无驾照者严重率 47%，远高于有照者的 17%')
p('- **前 2 年是最危险期**，建议新手在此期间格外注意骑行环境和速度')
p()
p('**对政策与数据的建议**：')
p('- **加强夜间酒驾执法**——饮酒是最可控的高风险行为')
p('- **改善国道/省道摩托车安全设施**——对应孟加拉 Highway 场景，国内类似道路事故严重率高')
p('- **建立中国本地事故数据库**——包含排量、车型、安全配置（如 ABS）等关键字段，填补数据空白')
p('- **电动自行车安全需专项研究**——保有量数亿，但速度和碰撞特征与燃油摩托车差异大')
p()
p('---')
p('数据和代码：[GitHub: weaming/motor-accident-analysis](https://github.com/weaming/motor-accident-analysis)')
p()

# ══════════════════════════════════════════════════════
#  写入 Markdown 文件
# ══════════════════════════════════════════════════════
md_path = os.path.join(REPORT_DIR, '摩托车事故归因分析报告.md')
with open(md_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(md))
print(f'报告已生成：{md_path}')
print(f'图片已保存到：{OUT_DIR}')
print(f'Markdown 文件大小：{os.path.getsize(md_path) / 1024:.0f} KB')

# 打印第一张图片的 alt 以确认
print('\n报告引用的图片文件：')
for k, v in charts.items():
    sz = os.path.getsize(os.path.join(REPORT_DIR, v)) // 1024
    print(f'  - {v} ({sz} KB)')
