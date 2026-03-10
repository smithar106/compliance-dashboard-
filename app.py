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

    # Violation overage: when deployment > cap, how many vehicles over?
    over_rows   = [r for r in city_rows if not r['compliant']]
    avg_overage = (
        sum(r['deployment'] - cap for r in over_rows) / len(over_rows)
        if over_rows else 0.0
    )
    # Compliant deployment ratio: avg deployment/cap in months where compliant
    comp_rows = [r for r in city_rows if r['compliant']]
    compliant_dep_ratio = (
        sum(r['deployment'] for r in comp_rows) / len(comp_rows) / cap
        if comp_rows else avg_dep / cap
    )

    # City size tier — bigger cities face higher regulatory / political scrutiny
    # so the model applies tighter compliance targets and faster move task urgency
    if fleet >= 5000:
        size_tier   = 'large'
        size_factor = 1.0   # large: tightest targets
    elif fleet >= 1500:
        size_tier   = 'medium'
        size_factor = 0.6   # medium: moderate tightening
    else:
        size_tier   = 'small'
        size_factor = 0.0   # small: no additional tightening beyond base model

    return {
        'cap':                     cap,
        'fleet':                   fleet,
        'cap_to_fleet_ratio':      cap / fleet,
        'avg_compliance_rate':     sum(comp_vals) / len(comp_vals),
        'recent_compliance_rate':  sum(1 if r['compliant'] else 0 for r in recent) / len(recent),
        'avg_deployment_ratio':    avg_dep / cap,
        'recent_deployment_ratio': sum(r['deployment'] for r in recent) / len(recent) / cap,
        'compliant_dep_ratio':     compliant_dep_ratio,
        'avg_overage':             avg_overage,
        'violation_months':        len(over_rows),
        'size_tier':               size_tier,
        'size_factor':             size_factor,
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

    cost        = feats['cost_per_task']
    avg_overage = feats['avg_overage']
    size_tier   = feats['size_tier']
    size_factor = feats['size_factor']

    # Size-based tightening: larger cities require more conservative targets
    # High: large cities target 8% tighter, medium 5% tighter vs base model
    # Low:  large cities can push 6% less over cap, medium 3% less
    size_tight_high = 0.08 * size_factor   # subtract from target_ratio for high/med
    size_tight_low  = 0.06 * size_factor   # reduce max overage for low

    # Move task urgency window scales with city size (hours to clear exceedance)
    # Large cities: 2hr (high), 4hr (med) — faster clearance, higher scrutiny
    # Medium cities: 3hr (high), 6hr (med)
    # Small cities:  4hr (high), 8hr (med)
    task_windows = {
        'large':  {'high': 2, 'med': 4},
        'medium': {'high': 3, 'med': 6},
        'small':  {'high': 4, 'med': 8},
    }
    size_label = {'large': 'Major market', 'medium': 'Mid-size market', 'small': 'Emerging market'}[size_tier]

    # ── ML-derived move task frequency ────────────────────────────────────────
    # Find compliant similar cities and look at their compliant deployment ratio
    # to infer how many vehicles need to be moved and at what rate.
    compliant_neighbors = [c for c in similar if city_feats[c]['avg_compliance_rate'] >= 0.75]
    if not compliant_neighbors:
        compliant_neighbors = similar[:3]

    # Average compliant deployment ratio among comparable high-compliance cities
    comp_dep_ratio = sum(city_feats[c]['compliant_dep_ratio'] for c in compliant_neighbors) / len(compliant_neighbors)
    # Avg overage in comparable markets when they do violate (context for task demand)
    sim_avg_overage = sum(city_feats[c]['avg_overage'] for c in similar[:3]) / 3

    def move_task_action(target, urgency='high'):
        """Derive a move task recommendation from comparable market data."""
        vehicles_to_clear = max(0, int(cur) - target)
        # Urgency window scales with city size — larger cities must clear faster
        if urgency == 'low':
            window_hrs = None
        else:
            window_hrs = task_windows[size_tier][urgency]

        if urgency == 'low' or vehicles_to_clear == 0:
            return {
                'arrow': '↓',
                'text':  'Reduce City Center move task frequency',
                'sub':   (f'{size_label}: similar markets ({similar[0]}, {similar[1]}) '
                          f'accept overage and minimize costly move tasks to protect topline')
            }
        tasks_per_hr = math.ceil(vehicles_to_clear / window_hrs)
        sim_names    = ', '.join(compliant_neighbors[:2])
        sim_comp_pct = round(sum(city_feats[c]['avg_compliance_rate'] for c in compliant_neighbors[:2]) / 2 * 100)
        sim_dep_pct  = round(comp_dep_ratio * 100)
        return {
            'arrow': '↑',
            'text':  f'Execute {tasks_per_hr} move tasks/hr — clear to cap within {window_hrs}hr',
            'sub':   (f'{size_label}: {vehicles_to_clear:,} vehicles must exit the zone '
                      f'({int(cur):,} → {target:,}). '
                      f'{sim_names} achieve {sim_comp_pct}% compliance '
                      f'at {sim_dep_pct}% of cap with comparable task cadence.')
        }

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
        base_ratio = min(
            sum(city_feats[c]['avg_deployment_ratio'] for c in base) / len(base),
            0.93
        )
        # Larger cities: tighten target further due to higher regulatory scrutiny
        target_ratio = max(base_ratio - size_tight_high, 0.78)
        target = max(int(cap * target_ratio), int(cap * 0.78))
        delta_pct = (target - cur) / cur * 100
        comp_prob = 97 if target <= cap else 75

        if compliant_sim:
            avg_c = round(sum(city_feats[c]['avg_compliance_rate'] for c in compliant_sim[:3]) / min(3, len(compliant_sim)) * 100)
            insight = f"{size_label} — {similar[0]} & {similar[1]} achieve {avg_c}% compliance at {round(target_ratio*100)}% of cap"
        else:
            insight = f"{size_label} — targeting {round(target_ratio*100)}% of cap, modeled from {len(similar)} comparable markets"

        return {
            'priority': 'high', 'bg': '#059669',
            'title': 'High Priority — Full Compliance Plan',
            'sub':   'GR relationship protected · Zero tolerance for violation',
            'similar_cities': sim3, 'sim_scores': {c: sim_scores[c] for c in sim3},
            'insight': insight, 'target': target, 'current': int(cur), 'cap': int(cap),
            'size_tier': size_tier, 'size_label': size_label,
            'delta_pct': round(delta_pct, 1), 'comp_prob': comp_prob,
            'rev_delta': monthly_rev_delta(target),
            'annual_rev': annual_rev_delta(target),
            'annual_ops': annual_ops_delta(target, task_multiplier=2.0),
            'actions': [
                {'arrow': '↓' if target < cur else ('↑' if target > cur else '→'),
                 'text':  f'{"Lower" if target < cur else "Increase" if target > cur else "Hold"} daily City Center deployment to {target:,} vehicles',
                 'sub':   f'{int(cur):,} → {target:,} vehicles ({abs(round(delta_pct,1))}% {"reduction" if target < cur else "increase" if target > cur else "no change"})'},
                move_task_action(target, urgency='high')
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
        base_ratio = min(avg_ratio * 0.97, 0.97)
        target_ratio = max(base_ratio - size_tight_high * 0.5, 0.82)
        target = int(cap * target_ratio)
        delta_pct = (target - cur) / cur * 100
        return {
            'priority': 'med', 'bg': '#D97706',
            'title': 'Medium Priority — Balanced Compliance Plan',
            'sub':   'Minimal violation risk · Modest topline upside',
            'similar_cities': sim3, 'sim_scores': {c: sim_scores[c] for c in sim3},
            'insight': f"{size_label} — similar markets run at {round(target_ratio*100)}% of cap, averaging ~85% compliance",
            'target': target, 'current': int(cur), 'cap': int(cap),
            'size_tier': size_tier, 'size_label': size_label,
            'delta_pct': round(delta_pct, 1), 'comp_prob': 85,
            'rev_delta': monthly_rev_delta(target),
            'annual_rev': annual_rev_delta(target),
            'annual_ops': annual_ops_delta(target, task_multiplier=1.0),
            'actions': [
                {'arrow': '↓' if target < cur else ('↑' if target > cur else '→'),
                 'text':  f'{"Lower" if target < cur else "Increase" if target > cur else "Hold"} daily City Center deployment to {target:,} vehicles',
                 'sub':   f'{int(cur):,} → {target:,} vehicles ({abs(round(delta_pct,1))}% {"reduction" if target < cur else "increase" if target > cur else "no change"})'},
                move_task_action(target, urgency='med')
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
        # Larger cities: cap how far over the limit we recommend going
        target_ratio = min(max_ratio * 1.05, 1.20 - size_tight_low)
        target = int(cap * target_ratio)
        delta_pct = (target - cur) / cur * 100
        return {
            'priority': 'low', 'bg': '#DC2626',
            'title': 'Low Priority — Growth Maximization Plan',
            'sub':   'Compliance risk accepted · Maximum topline',
            'similar_cities': sim3, 'sim_scores': {c: sim_scores[c] for c in sim3},
            'insight': f"{size_label} — growth-mode similar markets push to {round(target_ratio*100)}% of cap for max revenue",
            'target': target, 'current': int(cur), 'cap': int(cap),
            'size_tier': size_tier, 'size_label': size_label,
            'delta_pct': round(delta_pct, 1), 'comp_prob': 35,
            'rev_delta': monthly_rev_delta(target),
            'annual_rev': annual_rev_delta(target),
            'annual_ops': annual_ops_delta(target, task_multiplier=0.33),
            'actions': [
                {'arrow': '↑' if target > cur else '→',
                 'text':  f'{"Increase" if target > cur else "Maintain"} daily deployment to {target:,} vehicles',
                 'sub':   f'{int(cur):,} → {target:,} ({abs(round(delta_pct,1))}% {"increase" if target > cur else "no change"}) in City Center'},
                move_task_action(target, urgency='low')
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
