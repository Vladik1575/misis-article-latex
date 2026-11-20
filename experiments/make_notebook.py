"""Собирает starbucks_bandits.ipynb — эксперименты статьи. Запуск: .venv/bin/python make_notebook.py"""
import nbformat as nbf

cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip()))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip()))


md("""
# Бандиты по компонентам оффера на данных Starbucks

Эксперименты к статье «Персонализация маркетинговых коммуникаций…». Подход Aampe: сообщение — сочетание
критериев (тип оффера, порог, срок, каналы, слот времени); на каждый критерий свои веса, сочетание оценивается
суммой логитов, исследование — сэмплированием Томпсона.

1. **Данные.** Udacity / Starbucks rewards (симуляция Starbucks): 17 000 клиентов, 6 волн рассылки, в каждой ~75%
   клиентов получают один из 10 офферов случайно, ~25% — ничего. Награда — покупка за 72 ч после волны.
2. **Replay на логах.** Выбор среди 10 реальных офферов; несмещённая офлайн-оценка (Li et al., 2011).
3. **Полусинтетическая среда.** Отклик подогнан по данным, добавлен критерий «слот времени» с известным эффектом;
   288 сочетаний, считаем сожаление относительно оракула и прирост к «без оффера».

Код алгоритмов — `sblab.py`. Данные — `data/` (Kaggle `blacktile/starbucks-app-customer-reward-program-data`).
""")

code("""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from sblab import (CTX_NAMES, CriterionBeta, DisjointLogisticTS, FlatBeta, LabelLogisticTS, RandomPolicy,
                   TrueEnv, factorial_actions, load_data, logit, portfolio_actions, replay, simulate)

FIG = Path("../figures"); FIG.mkdir(exist_ok=True)
RESULTS = {}

# Цвет закреплён за политикой (не за рангом); в печати различаем ещё и типом линии
STYLE = {
    "Случайная":                          dict(color="#898989", ls=":"),
    "Beta на сочетание":                  dict(color="#2a78d6", ls="--"),
    "Бандит на критерий":                 dict(color="#eb6834", ls="-"),
    "Логистика по компонентам":           dict(color="#1baf7a", ls="-."),
    "Логистика по компонентам + контекст": dict(color="#6250d6", ls="-"),
    "Логистика на сочетание + контекст":  dict(color="#e87ba4", ls="--"),
}
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6,
                     "lines.linewidth": 1.6, "figure.dpi": 150, "savefig.bbox": "tight"})
""")

md("## 1. Данные: офферы, отклик и проверка случайности рассылки")

code("""
po, users, waves = load_data("data")
port = portfolio_actions(po)
X = users[CTX_NAMES].to_numpy()

def rate_ci(y):
    p = y.mean(); return p, 1.96 * np.sqrt(p * (1 - p) / len(y))

p0, _ = rate_ci(waves[waves.offer < 0].buy)
rows = []
for i, r in po.iterrows():
    y = waves[waves.offer == i].buy
    p, h = rate_ci(y)
    rows.append(dict(offer=i, тип=r.offer_type, награда=r.reward, порог=r.difficulty, срок=r.duration,
                     каналы=",".join(sorted(r.channels)), n=len(y), покупка_72ч=round(p, 3),
                     прирост=round(p - p0, 3), ci95=round(h, 3)))
offers = pd.DataFrame(rows).sort_values("покупка_72ч", ascending=False)
n0 = (waves.offer < 0).sum()
print(f"без оффера: n={n0}, покупка за 72 ч = {p0:.3f}")
offers
""")

code("""
# Случайность рассылки: оффер (включая «без оффера») не зависит от признаков клиента
m = waves.merge(users, left_on="uid", right_index=True)
m["возраст"] = pd.cut(m.age, [-9, -1, 0, 1, 9]).astype(str)
m["доход"] = pd.cut(m.income, [-9, -0.7, 0, 0.7, 9]).astype(str)
m["пол"] = np.select([m.female == 1, m.male == 1, m.no_profile == 1], ["F", "M", "нет"], "O")
balance = {c: chi2_contingency(pd.crosstab(m.offer, m[c]))[1] for c in ["пол", "возраст", "доход"]}
print("p-value χ² (оффер × признак):", {k: round(v, 3) for k, v in balance.items()})
RESULTS["data"] = dict(n_users=len(users), n_waves=int(waves.wave.nunique()), n_offer_events=int((waves.offer >= 0).sum()),
                       n_none=int(n0), p_none=float(p0), offers=offers.to_dict("records"),
                       balance_p={k: float(v) for k, v in balance.items()})
""")

md("""
## 2. Replay на реальных логах: выбор среди 10 офферов

Условно на факт отправки оффер выбирался равномерно (1/10), поэтому replay даёт несмещённую оценку
средней награды политики. Каждая политика проходит логи в случайном порядке 30 раз.
""")

code("""
off = waves[waves.offer >= 0]
Xo, ao, yo = X[off.uid.to_numpy()], off.offer.to_numpy(), off.buy.to_numpy(float)
b0 = logit(yo.mean())   # априорное среднее свободного члена — средний отклик на оффер

def replay_policies():
    return [RandomPolicy(port), FlatBeta(port), CriterionBeta(port),
            LabelLogisticTS(port, len(CTX_NAMES), contextual=False, b0=b0),
            LabelLogisticTS(port, len(CTX_NAMES), contextual=True, b0=b0),
            DisjointLogisticTS(port, len(CTX_NAMES), b0=b0)]

SEEDS, N_CURVE = 30, 7000
runs = {}
for seed in range(SEEDS):
    for pol in replay_policies():
        r = replay(pol, Xo, ao, yo, np.random.default_rng(seed))
        runs.setdefault(pol.label, []).append(r[:N_CURVE])
""")

code("""
def mean_ci(v):
    v = np.asarray(v); return v.mean(), 1.96 * v.std(ddof=1) / np.sqrt(len(v))

best_offer = offers.покупка_72ч.max()
rows = []
for lab, rs in runs.items():
    R = np.vstack(rs)
    first, last, total = R[:, :1000].mean(1), R[:, -2000:].mean(1), R.mean(1)
    rows.append(dict(политика=lab, первые_1000=mean_ci(first)[0], последние_2000=mean_ci(last)[0],
                     все=mean_ci(total)[0], ci95=mean_ci(total)[1], прирост_к_без_оффера=mean_ci(total)[0] - p0))
replay_tab = pd.DataFrame(rows).round(4)
print(f"лучший оффер в среднем по логам: {best_offer:.3f}; без оффера: {p0:.3f}")
RESULTS["replay"] = dict(seeds=SEEDS, n_curve=N_CURVE, best_offer=float(best_offer), table=replay_tab.to_dict("records"))
replay_tab
""")

code("""
fig, ax = plt.subplots(figsize=(3.4, 2.3))
n = np.arange(1, N_CURVE + 1)
for lab, rs in runs.items():
    curve = (np.cumsum(np.vstack(rs), axis=1) / n).mean(0)
    ax.plot(n[100:], curve[100:], label=lab, **STYLE[lab])
ax.axhline(best_offer, color="#555", lw=0.8)
ax.text(N_CURVE, best_offer + 0.002, "лучший оффер", ha="right", va="bottom", fontsize=7, color="#555")
ax.set_xlabel("принятые события replay"); ax.set_ylabel("доля покупок за 72 ч\\n(нарастающее среднее)")
ax.legend(fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.28), ncol=2)
fig.savefig(FIG / "replay.pdf"); plt.show()
""")

md("""
## 3. Полусинтетическая среда: 288 сочетаний

Истинный отклик — логистика `клиент × компоненты`, подогнанная по всем волнам (в том числе без оффера).
Критерий «слот» (утро / день / вечер / ночь) в данных отсутствует — его эффект задан симулятором: сдвиг на слот
и наклон по возрасту (молодым лучше вечер, старшим — утро). Каждый день 500 случайных клиентов, обновление в конце дня.
""")

code("""
env = TrueEnv(po, users, waves)
fa = factorial_actions(with_slot=True)
fa72 = factorial_actions(with_slot=False)
eff = pd.DataFrame({"признак": env.feat_names, "вес": env.w, "sd": env.w_sd})
show = eff[eff.признак.isin(port.feat_names) | eff.признак.str.startswith("sent")].round(3)
P = env.prob(X, fa)
print(f"сочетаний: {fa.n}; средний отклик сочетаний {P.mean(0).min():.3f}…{P.mean(0).max():.3f}; "
      f"оракул {P.max(1).mean():.3f}; без оффера {env.prob_none(X).mean():.3f}")
RESULTS["env"] = dict(n_actions=fa.n, p_min=float(P.mean(0).min()), p_max=float(P.mean(0).max()),
                      p_oracle=float(P.max(1).mean()), p_none=float(env.prob_none(X).mean()),
                      oracle_distinct=int((np.bincount(P.argmax(1), minlength=fa.n) > 0).sum()),
                      slot_base=env.SLOT_BASE.tolist(), slot_age=env.SLOT_AGE.tolist())
show
""")

code("""
DAYS, PER_DAY, SIM_SEEDS = 60, 500, 20

def sim_policies(acts):
    return [RandomPolicy(acts), FlatBeta(acts), CriterionBeta(acts),
            LabelLogisticTS(acts, len(CTX_NAMES), contextual=False, b0=b0),
            LabelLogisticTS(acts, len(CTX_NAMES), contextual=True, b0=b0)]

def run_sim(acts, per_day=PER_DAY):
    out = {}
    for seed in range(SIM_SEEDS):
        for pol in sim_policies(acts):
            reg, val, up = simulate(pol, env, acts, X, DAYS, per_day, np.random.default_rng(100 + seed))
            out.setdefault(pol.label, []).append((reg, val, up))
    return out

sims = {"288": run_sim(fa), "72": run_sim(fa72)}
""")

code("""
def sim_table(out, acts):
    Pm = env.prob(X, acts)
    up_oracle = (Pm.max(1) - env.prob_none(X)).mean()
    rows = []
    for lab, rs in out.items():
        reg = np.array([r[0].sum() for r in rs]) / PER_DAY            # сожаление на клиента-день, сумма за горизонт
        up_last = np.array([r[2][-10:].mean() for r in rs])
        rows.append(dict(политика=lab, сожаление=mean_ci(reg)[0], ci95=mean_ci(reg)[1],
                         прирост_последние_10_дней=up_last.mean(), доля_прироста_оракула=up_last.mean() / up_oracle))
    return pd.DataFrame(rows).round(4), up_oracle

tabs = {}
for k, acts in [("288", fa), ("72", fa72)]:
    tabs[k], upo = sim_table(sims[k], acts)
    RESULTS[f"sim_{k}"] = dict(days=DAYS, per_day=PER_DAY, seeds=SIM_SEEDS, uplift_oracle=float(upo),
                               table=tabs[k].to_dict("records"))
    print(f"--- {k} сочетаний, прирост оракула к «без оффера» = {upo:.3f}"); display(tabs[k])
""")

code("""
fig, ax = plt.subplots(figsize=(3.4, 2.3))
days = np.arange(1, DAYS + 1)
for lab, rs in sims["288"].items():
    cum = np.mean([np.cumsum(r[0]) / PER_DAY for r in rs], axis=0)
    ax.plot(days, cum, label=lab, **STYLE[lab])
ax.set_xlabel("день"); ax.set_ylabel("накопленное сожаление\\nна клиента")
ax.legend(fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.28), ncol=2)
fig.savefig(FIG / "sim_regret.pdf"); plt.show()
""")

md("""
## 4. Что выучила контекстная модель: эффект слота по возрасту

Сравниваем апостериорное среднее эффекта слота (относительно утра) у младших и старших клиентов с истиной симулятора.
""")

code("""
pol = LabelLogisticTS(fa, len(CTX_NAMES), contextual=True, b0=b0)
simulate(pol, env, fa, X, DAYS, PER_DAY, np.random.default_rng(7))
slot_cols = [fa.feat_names.index(f"slot={s}") for s in ["day", "evening", "night"]]
age = X[:, CTX_NAMES.index("age")]
rows = []
for name, mask in [("моложе (age_z < −0.7)", (age < -0.7) & (X[:, 1] == 0)), ("старше (age_z > 0.7)", age > 0.7)]:
    est = pol.mean_effects(X[mask])[:, slot_cols].mean(0)
    a = age[mask].mean()
    true = env.SLOT_BASE[1:] + a * env.SLOT_AGE[1:] - (env.SLOT_BASE[0] + a * env.SLOT_AGE[0])
    rows.append(dict(группа=name, **{f"{s}: оценка": e for s, e in zip(["день", "вечер", "ночь"], est)},
                     **{f"{s}: истина": t for s, t in zip(["день", "вечер", "ночь"], true)}))
slot_tab = pd.DataFrame(rows).round(2)
RESULTS["slot_recovery"] = slot_tab.to_dict("records")
slot_tab
""")

code("""
Path("results.json").write_text(json.dumps(RESULTS, ensure_ascii=False, indent=1, default=float))
print("сохранено: results.json, ../figures/replay.pdf, ../figures/sim_regret.pdf")
""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "starbucks_bandits.ipynb")
print("starbucks_bandits.ipynb:", len(cells), "ячеек")
