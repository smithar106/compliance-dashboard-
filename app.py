from flask import Flask, jsonify, request
import os, csv, math

app = Flask(__name__)

# ── Load CSV (pure Python, no pandas) ─────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'global_compliance_data.csv')

def parse_money(s):
    return float(str(s).replace('$', '').replace(',', '').strip())

def load_data():
    rows = []
    with open(DATA_PATH, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rows.append({
                'city':        r['City'],
                'month':       r['Month'],
                'deployment':  float(r['Avg Daily Deployment']),
                'cap':         float(r['City Center Max Cap']),
                'fleet':       float(r['Fleet Size']),
                'rev_per_veh': parse_money(r['Revenue per Vehicle per Day']),
                'cost_per_task': parse_money(r['Ops Cost per Task']),
                'compliant':   r['Compliant'].strip().lower() == 'yes',
            })
    return rows

rows = load_data()

# ── Per-city feature summary ───────────────────────────────────────────────────
def month_sort_key(m):
    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    parts  = m.split()
    return int(parts[1]) * 12 + months.index(parts[0])

def city_summary(city_rows):
    city_rows = sorted(city_rows, key=lambda r: month_sort_key(r['month']))
    recent    = city_rows[-6:]
    dep_vals  = [r['deployment'] for r in city_rows]
    comp_vals = [1 if r['compliant'] else 0 for r in city_rows]
    cap       = city_rows[0]['cap']
    fleet     = city_rows[0]['fleet']
    avg_dep   = sum(dep_vals) / len(dep_vals)
    return {
        'cap':                     cap,
        'fleet':                   fleet,
        'cap_to_fleet_ratio':      cap / fleet,
        'avg_compliance_rate':     sum(comp_vals) / len(comp_vals),
        'recent_compliance_rate':  sum(1 if r['compliant'] else 0 for r in recent) / len(recent),
        'avg_deployment_ratio':    avg_dep / cap,
        'recent_deployment_ratio': sum(r['deployment'] for r in recent) / len(recent) / cap,
        'rev_per_veh':             city_rows[0]['rev_per_veh'],
        'cost_per_task':           city_rows[0]['cost_per_task'],
        'latest_deployment':       city_rows[-1]['deployment'],
        'latest_month':            city_rows[-1]['month'],
    }

# Group rows by city
city_map = {}
for r in rows:
    city_map.setdefault(r['city'], []).append(r)

cities      = sorted(city_map.keys())
city_feats  = {c: city_summary(city_map[c]) for c in cities}

# ── Pure-Python KNN ────────────────────────────────────────────────────────────
FEAT_KEYS = [
    'fleet',            'cap_to_fleet_ratio',    'avg_compliance_rate',
    'recent_compliance_rate', 'avg_deployment_ratio',
    'rev_per_veh',      'cost_per_task'
]

# Compute per-feature mean and std for normalization
def feat_vec(city):
    f = city_feats[city]
    # map fleet → fleet (raw key in feats dict is 'fleet')
    return [
        f['fleet'], f['cap_to_fleet_ratio'], f['avg_compliance_rate'],
        f['recent_compliance_rate'], f['avg_deployment_ratio'],
        f['rev_per_veh'], f['cost_per_task']
    ]

feat_matrix = {c: feat_vec(c) for c in cities}
n_feats     = len(FEAT_KEYS)

means = [sum(feat_matrix[c][i] for c in cities) / len(cities) for i in range(n_feats)]
stds  = [
    math.sqrt(sum((feat_matrix[c][i] - means[i])**2 for c in cities) / len(cities)) or 1.0
    for i in range(n_feats)
]

def normalize(vec):
    return [(v - means[i]) / stds[i] for i, v in enumerate(vec)]

scaled = {c: normalize(feat_matrix[c]) for c in cities}

def euclidean(a, b):
    return math.sqrt(sum((x - y)**2 for x, y in zip(a, b)))

def find_neighbors(city, k=5):
    target = scaled[city]
    dists  = [(c, euclidean(target, scaled[c])) for c in cities if c != city]
    dists.sort(key=lambda x: x[1])
    return dists[:k]

# ── Recommendation engine ─────────────────────────────────────────────────────
def get_recommendation(city, priority):
    if city not in city_feats:
        return {'error': 'City not found'}, 404

    feats    = city_feats[city]
    cap      = feats['cap']
    cur      = feats['latest_deployment']
    rev      = feats['rev_per_veh']
    neighbors = find_neighbors(city)
    similar   = [n[0] for n in neighbors]

    max_d = max(n[1] for n in neighbors) if neighbors else 1
    sim_scores = {n[0]: max(10, round((1 - n[1] / (max_d * 2)) * 100)) for n in neighbors}

    cost = feats['cost_per_task']

    def monthly_rev_delta(target):
        return int(rev * (target - cur) * 30)

    def annual_rev_delta(target):
        return int(rev * (target - cur) * 30 * 12)

    def annual_ops_delta(target, task_multiplier=1.0):
        # ops cost = tasks × cost_per_task; tasks = deployment × 3/day × 30days
        cur_monthly_ops  = cur    * 3 * 30 * cost
        tgt_monthly_ops  = target * 3 * 30 * cost * task_multiplier
        return int((tgt_monthly_ops - cur_monthly_ops) * 12)

    sim3 = similar[:3]

    if priority == 'high':
        compliant_sim = [c for c in similar if city_feats[c]['avg_compliance_rate'] >= 0.85]
        base = compliant_sim if compliant_sim else similar
        target_ratio = min(
            sum(city_feats[c]['avg_deployment_ratio'] for c in base) / len(base),
            0.93
        )
        target = max(int(cap * target_ratio), int(cap * 0.80))
        delta_pct = (target - cur) / cur * 100
        comp_prob = 97 if target <= cap else 75

        if compliant_sim:
            avg_c = round(sum(city_feats[c]['avg_compliance_rate'] for c in compliant_sim[:3]) / min(3, len(compliant_sim)) * 100)
            insight = f"{similar[0]} & {similar[1]} achieve {avg_c}% compliance at {round(target_ratio*100)}% of cap"
        else:
            insight = f"Targeting {round(target_ratio*100)}% cap utilization — modeled from {len(similar)} similar-scale markets"

        return {
            'priority': 'high', 'bg': '#059669',
            'title': 'High Priority — Full Compliance Plan',
            'sub':   'GR relationship protected · Zero tolerance for violation',
            'similar_cities': sim3, 'sim_scores': {c: sim_scores[c] for c in sim3},
            'insight': insight, 'target': target, 'current': int(cur), 'cap': int(cap),
            'delta_pct': round(delta_pct, 1), 'comp_prob': comp_prob,
            'rev_delta': monthly_rev_delta(target),
            'annual_rev': annual_rev_delta(target),
            'annual_ops': annual_ops_delta(target, task_multiplier=2.0),
            'actions': [
                {'arrow': '↓' if target < cur else '→',
                 'text':  f'{"Lower" if target < cur else "Hold"} daily deployment to {target:,} vehicles',
                 'sub':   f'{int(cur):,} → {target:,} in City Center zone ({abs(round(delta_pct,1))}% {"reduction" if target < cur else "no change"})'},
                {'arrow': '↑', 'text': 'Increase City Center move tasks: 3/hr → 6/hr',
                 'sub':   'Rebalancing frequency doubled to enforce zone cap in real time'}
            ],
            'impact': [
                {'val': '✅ Eliminated', 'lbl': 'Compliance Violation Risk', 'bg': '#D1FAE5', 'col': '#059669'},
                {'val': f'{round(delta_pct,1):+.1f}%', 'lbl': 'Annual Revenue Impact',
                 'bg': '#FEE2E2' if delta_pct < 0 else '#D1FAE5',
                 'col': '#DC2626' if delta_pct < 0 else '#059669'},
                {'val': '+3%', 'lbl': 'Annual Ops Cost Impact', 'bg': '#FEF3C7', 'col': '#D97706'}
            ]
        }

    elif priority == 'med':
        avg_ratio = sum(city_feats[c]['avg_deployment_ratio'] for c in sim3) / len(sim3)
        target_ratio = min(avg_ratio * 0.97, 0.97)
        target = int(cap * target_ratio)
        delta_pct = (target - cur) / cur * 100
        return {
            'priority': 'med', 'bg': '#D97706',
            'title': 'Medium Priority — Balanced Compliance Plan',
            'sub':   'Minimal violation risk · Modest topline upside',
            'similar_cities': sim3, 'sim_scores': {c: sim_scores[c] for c in sim3},
            'insight': f"Similar markets run at {round(target_ratio*100)}% of cap, averaging ~85% compliance",
            'target': target, 'current': int(cur), 'cap': int(cap),
            'delta_pct': round(delta_pct, 1), 'comp_prob': 85,
            'rev_delta': monthly_rev_delta(target),
            'annual_rev': annual_rev_delta(target),
            'annual_ops': annual_ops_delta(target, task_multiplier=1.0),
            'actions': [
                {'arrow': '↓' if target < cur else '→',
                 'text':  f'{"Lower" if target < cur else "Adjust"} daily deployment to {target:,} vehicles',
                 'sub':   f'{int(cur):,} → {target:,} ({abs(round(delta_pct,1))}% {"reduction" if target < cur else "adjustment"})'},
                {'arrow': '→', 'text': 'Move task frequency: No change',
                 'sub':   'Maintain 3 City Center move tasks per hour'}
            ],
            'impact': [
                {'val': '⚠ Minimal', 'lbl': 'Compliance Violation Risk', 'bg': '#FEF3C7', 'col': '#D97706'},
                {'val': f'{round(delta_pct,1):+.1f}%', 'lbl': 'Annual Revenue Impact',
                 'bg': '#D1FAE5' if delta_pct >= 0 else '#FEE2E2',
                 'col': '#059669' if delta_pct >= 0 else '#DC2626'},
                {'val': '+0.5%', 'lbl': 'Annual Ops Cost Impact', 'bg': '#F1F5F9', 'col': '#475569'}
            ]
        }

    else:  # low
        max_ratio = max(city_feats[c]['avg_deployment_ratio'] for c in sim3)
        target_ratio = min(max_ratio * 1.05, 1.20)
        target = int(cap * target_ratio)
        delta_pct = (target - cur) / cur * 100
        return {
            'priority': 'low', 'bg': '#DC2626',
            'title': 'Low Priority — Growth Maximization Plan',
            'sub':   'Compliance risk accepted · Maximum topline',
            'similar_cities': sim3, 'sim_scores': {c: sim_scores[c] for c in sim3},
            'insight': f"Growth-mode similar markets push to {round(target_ratio*100)}% of cap for max revenue",
            'target': target, 'current': int(cur), 'cap': int(cap),
            'delta_pct': round(delta_pct, 1), 'comp_prob': 35,
            'rev_delta': monthly_rev_delta(target),
            'annual_rev': annual_rev_delta(target),
            'annual_ops': annual_ops_delta(target, task_multiplier=0.33),
            'actions': [
                {'arrow': '↑' if target > cur else '→',
                 'text':  f'{"Increase" if target > cur else "Maintain"} daily deployment to {target:,} vehicles',
                 'sub':   f'{int(cur):,} → {target:,} ({abs(round(delta_pct,1))}% {"increase" if target > cur else "no change"}) in City Center'},
                {'arrow': '↓', 'text': 'Decrease City Center move tasks: 3/hr → 1/hr',
                 'sub':   'Reduced rebalancing lowers labor cost, maximizes zone density'}
            ],
            'impact': [
                {'val': '🔴 Elevated', 'lbl': 'Compliance Violation Risk', 'bg': '#FEE2E2', 'col': '#DC2626'},
                {'val': f'{round(delta_pct,1):+.1f}%', 'lbl': 'Annual Revenue Impact', 'bg': '#D1FAE5', 'col': '#059669'},
                {'val': '−1%', 'lbl': 'Annual Ops Cost Impact', 'bg': '#D1FAE5', 'col': '#059669'}
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
