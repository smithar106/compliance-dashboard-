from flask import Flask, jsonify, request
import os, json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

# ── Data loading ───────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'global_compliance_data.csv')

def parse_money(s):
    if isinstance(s, (int, float)):
        return float(s)
    return float(str(s).replace('$', '').replace(',', '').strip())

def load_data():
    df = pd.read_csv(DATA_PATH)
    for col in ['Revenue per Month', 'Ops Cost per Month', 'Ops Profit per Month',
                'Revenue per Vehicle per Day', 'Ops Cost per Task']:
        df[col] = df[col].apply(parse_money)
    df['compliant_bin'] = (df['Compliant'].str.lower() == 'yes').astype(int)
    df['month_dt'] = pd.to_datetime(df['Month'], format='%b %Y')
    df = df.sort_values(['City', 'month_dt'])
    return df

df_raw = load_data()

# ── Per-city feature summary ───────────────────────────────────────────────────
def city_features(df, city):
    c = df[df['City'] == city].copy()
    recent = c.tail(6)
    cap = c['City Center Max Cap'].iloc[0]
    fleet = c['Fleet Size'].iloc[0]
    avg_dep = c['Avg Daily Deployment'].mean()
    latest_dep = c['Avg Daily Deployment'].iloc[-1]
    avg_comp = c['compliant_bin'].mean()
    recent_comp = recent['compliant_bin'].mean()
    return {
        'fleet_size':              float(fleet),
        'cap':                     float(cap),
        'cap_to_fleet_ratio':      cap / fleet,
        'avg_compliance_rate':     avg_comp,
        'recent_compliance_rate':  recent_comp,
        'avg_deployment_ratio':    avg_dep / cap,
        'recent_deployment_ratio': recent['Avg Daily Deployment'].mean() / cap,
        'revenue_per_veh_day':     float(c['Revenue per Vehicle per Day'].iloc[0]),
        'ops_cost_per_task':       float(c['Ops Cost per Task'].iloc[0]),
        'avg_daily_deployment':    avg_dep,
        'latest_deployment':       float(latest_dep),
        'latest_month':            c['Month'].iloc[-1],
    }

cities    = sorted(df_raw['City'].unique().tolist())
city_feats = {city: city_features(df_raw, city) for city in cities}

# ── KNN model ─────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    'fleet_size', 'cap_to_fleet_ratio', 'avg_compliance_rate',
    'recent_compliance_rate', 'avg_deployment_ratio',
    'revenue_per_veh_day', 'ops_cost_per_task'
]

city_list   = list(cities)
feat_matrix = np.array([[city_feats[c][f] for f in FEATURE_COLS] for c in city_list])
scaler      = StandardScaler()
feat_scaled = scaler.fit_transform(feat_matrix)

knn = NearestNeighbors(n_neighbors=min(6, len(city_list)), metric='euclidean')
knn.fit(feat_scaled)

# ── Recommendation engine ─────────────────────────────────────────────────────
def get_recommendation(city, priority):
    if city not in city_feats:
        return {'error': 'City not found'}, 404

    feats = city_feats[city]
    cap   = feats['cap']
    cur   = feats['latest_deployment']
    rev   = feats['revenue_per_veh_day']

    # Find KNN neighbors (exclude self)
    query        = np.array([[feats[f] for f in FEATURE_COLS]])
    query_scaled = scaler.transform(query)
    distances, indices = knn.kneighbors(query_scaled)
    neighbors    = [(city_list[i], distances[0][j])
                    for j, i in enumerate(indices[0]) if city_list[i] != city][:5]
    similar      = [n[0] for n in neighbors]
    sim_feats    = [city_feats[c] for c in similar]

    # Similarity scores (convert distance to 0-100 score)
    max_d = max(n[1] for n in neighbors) if neighbors else 1
    sim_scores = {n[0]: round((1 - n[1] / (max_d * 2)) * 100) for n in neighbors}

    def monthly_rev_delta(target):
        return int(rev * (target - cur) * 30)

    if priority == 'high':
        # Target: deployment ratio of compliant similar cities
        compliant_sim = [c for c in similar if city_feats[c]['avg_compliance_rate'] >= 0.85]
        base = compliant_sim if compliant_sim else similar
        target_ratio = min(np.mean([city_feats[c]['avg_deployment_ratio'] for c in base]), 0.93)
        target = max(int(cap * target_ratio), int(cap * 0.80))
        delta_pct = (target - cur) / cur * 100
        comp_prob = 97 if target <= cap else 75
        if compliant_sim:
            avg_sim_comp = round(np.mean([city_feats[c]['avg_compliance_rate'] for c in compliant_sim[:3]]) * 100)
            insight_txt = f"{similar[0]} & {similar[1]} achieve {avg_sim_comp}% compliance at {round(target_ratio*100)}% of cap"
        else:
            insight_txt = f"Targeting {round(target_ratio*100)}% cap utilization — modeled from {len(similar)} similar-scale markets"
        return {
            'priority':    'high',
            'title':       'High Priority — Full Compliance Plan',
            'sub':         'GR relationship protected · Zero tolerance for violation',
            'bg':          '#059669',
            'similar_cities': similar[:3],
            'sim_scores':  {c: sim_scores[c] for c in similar[:3]},
            'insight':     insight_txt,
            'target':      target,
            'current':     int(cur),
            'cap':         int(cap),
            'delta_pct':   round(delta_pct, 1),
            'comp_prob':   comp_prob,
            'rev_delta':   monthly_rev_delta(target),
            'actions': [
                {
                    'arrow': '↓' if target < cur else '→',
                    'text':  f'{"Lower" if target < cur else "Hold"} daily deployment to {target:,} vehicles',
                    'sub':   f'{int(cur):,} → {target:,} in City Center zone ({abs(round(delta_pct,1))}% {"reduction" if target < cur else "no change"})'
                },
                {
                    'arrow': '↑',
                    'text':  'Increase City Center move tasks: 3/hr → 6/hr',
                    'sub':   'Rebalancing frequency doubled to enforce zone cap in real time'
                }
            ],
            'impact': [
                {'val': '✅ Eliminated', 'lbl': 'Compliance Violation Risk', 'bg': '#D1FAE5', 'col': '#059669'},
                {'val': f'{round(delta_pct,1):+.1f}%', 'lbl': 'Topline Revenue Impact',
                 'bg': '#FEE2E2' if delta_pct < 0 else '#D1FAE5',
                 'col': '#DC2626' if delta_pct < 0 else '#059669'},
                {'val': '+3%', 'lbl': 'Labor Cost to Ops', 'bg': '#FEF3C7', 'col': '#D97706'}
            ]
        }

    elif priority == 'med':
        avg_ratio = np.mean([city_feats[c]['avg_deployment_ratio'] for c in similar[:3]])
        target_ratio = min(avg_ratio * 0.97, 0.97)
        target = int(cap * target_ratio)
        delta_pct = (target - cur) / cur * 100
        return {
            'priority':    'med',
            'title':       'Medium Priority — Balanced Compliance Plan',
            'sub':         'Minimal violation risk · Modest topline upside',
            'bg':          '#D97706',
            'similar_cities': similar[:3],
            'sim_scores':  {c: sim_scores[c] for c in similar[:3]},
            'insight':     f"Similar markets run at {round(target_ratio*100)}% of cap, averaging ~85% compliance",
            'target':      target,
            'current':     int(cur),
            'cap':         int(cap),
            'delta_pct':   round(delta_pct, 1),
            'comp_prob':   85,
            'rev_delta':   monthly_rev_delta(target),
            'actions': [
                {
                    'arrow': '↓' if target < cur else '→',
                    'text':  f'{"Lower" if target < cur else "Adjust"} daily deployment to {target:,} vehicles',
                    'sub':   f'{int(cur):,} → {target:,} ({abs(round(delta_pct,1))}% {"reduction" if target < cur else "adjustment"})'
                },
                {
                    'arrow': '→',
                    'text':  'Move task frequency: No change',
                    'sub':   'Maintain 3 City Center move tasks per hour'
                }
            ],
            'impact': [
                {'val': '⚠ Minimal', 'lbl': 'Compliance Violation Risk', 'bg': '#FEF3C7', 'col': '#D97706'},
                {'val': f'{round(delta_pct,1):+.1f}%', 'lbl': 'Topline Revenue Impact',
                 'bg': '#D1FAE5' if delta_pct >= 0 else '#FEE2E2',
                 'col': '#059669' if delta_pct >= 0 else '#DC2626'},
                {'val': '+0.5%', 'lbl': 'Labor Cost to Ops', 'bg': '#F1F5F9', 'col': '#475569'}
            ]
        }

    else:  # low
        max_ratio = max(city_feats[c]['avg_deployment_ratio'] for c in similar[:3])
        target_ratio = min(max_ratio * 1.05, 1.20)
        target = int(cap * target_ratio)
        delta_pct = (target - cur) / cur * 100
        return {
            'priority':    'low',
            'title':       'Low Priority — Growth Maximization Plan',
            'sub':         'Compliance risk accepted · Maximum topline',
            'bg':          '#DC2626',
            'similar_cities': similar[:3],
            'sim_scores':  {c: sim_scores[c] for c in similar[:3]},
            'insight':     f"Growth-mode similar markets push to {round(target_ratio*100)}% of cap for max revenue",
            'target':      target,
            'current':     int(cur),
            'cap':         int(cap),
            'delta_pct':   round(delta_pct, 1),
            'comp_prob':   35,
            'rev_delta':   monthly_rev_delta(target),
            'actions': [
                {
                    'arrow': '↑' if target > cur else '→',
                    'text':  f'{"Increase" if target > cur else "Maintain"} daily deployment to {target:,} vehicles',
                    'sub':   f'{int(cur):,} → {target:,} ({abs(round(delta_pct,1))}% {"increase" if target > cur else "no change"}) in City Center'
                },
                {
                    'arrow': '↓',
                    'text':  'Decrease City Center move tasks: 3/hr → 1/hr',
                    'sub':   'Reduced rebalancing lowers labor cost, maximizes zone density'
                }
            ],
            'impact': [
                {'val': '🔴 Elevated', 'lbl': 'Compliance Violation Risk', 'bg': '#FEE2E2', 'col': '#DC2626'},
                {'val': f'{round(delta_pct,1):+.1f}%', 'lbl': 'Topline Revenue Impact', 'bg': '#D1FAE5', 'col': '#059669'},
                {'val': '−1%', 'lbl': 'Labor Cost to Ops', 'bg': '#D1FAE5', 'col': '#059669'}
            ]
        }

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    with open(os.path.join(os.path.dirname(__file__), 'index.html')) as f:
        return f.read()

@app.route('/api/recommend')
def recommend():
    city     = request.args.get('city', '').strip()
    priority = request.args.get('priority', 'high').strip()
    if not city:
        return jsonify({'error': 'city parameter required'}), 400
    if priority not in ('high', 'med', 'low'):
        return jsonify({'error': 'priority must be high/med/low'}), 400
    result = get_recommendation(city, priority)
    if isinstance(result, tuple):
        return jsonify(result[0]), result[1]
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5010)))
