"""Эксперименты статьи: бандиты по компонентам оффера на данных Starbucks (подход Aampe).

Что здесь есть:
  * load_data        — portfolio / profile / transcript → таблица «клиент × волна рассылки»;
  * Actions          — множество действий как сочетаний критериев (тип, порог, срок, каналы, слот);
  * политики          — случайная, плоский Beta-бандит, бандит на каждый критерий с суммой логитов,
                        байесовская логистика по компонентам (без контекста и с контекстом),
                        логистика на каждое действие отдельно (disjoint);
  * replay           — несмещённая офлайн-оценка по логам со случайным выбором (Li et al., 2011);
  * TrueEnv          — полусинтетическая среда: отклик подогнан по данным, слот времени задан симулятором.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

WINDOW_H = 72  # окно отклика: покупка в течение 72 ч после волны (минимальный интервал между волнами)

# ---------------------------------------------------------------- данные

CTX_NAMES = ["const", "no_profile", "female", "male", "age", "income", "tenure"]


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def load_data(data_dir: str | Path):
    """Возвращает (portfolio, users, waves).

    portfolio — 10 офферов с компонентами; users — контекст клиента (7 признаков);
    waves — строка на клиента × волну: offer (индекс 0..9 или -1 = без оффера), buy (покупка за 72 ч).
    """
    d = Path(data_dir)
    po = pd.read_json(d / "portfolio.json", lines=True)
    pr = pd.read_json(d / "profile.json", lines=True)
    tr = pd.read_json(d / "transcript.json", lines=True)

    po = po.assign(
        bogo=(po.offer_type == "bogo").astype(int),
        discount=(po.offer_type == "discount").astype(int),
        hard=(po.difficulty >= 10).astype(int),
        long=(po.duration >= 7).astype(int),
        social=po.channels.apply(lambda c: int("social" in c)),
        web=po.channels.apply(lambda c: int("web" in c)),
        mobile=po.channels.apply(lambda c: int("mobile" in c)),
    ).reset_index(names="offer")

    # контекст клиента: пропуск профиля, пол, возраст, доход, стаж (z-оценки, у пропусков — 0)
    no_prof = (pr.age == 118).to_numpy()
    member = pd.to_datetime(pr.became_member_on.astype(str), format="%Y%m%d")
    tenure = (member.max() - member).dt.days.to_numpy(float)

    def z(v):
        v = np.asarray(v, float)
        ok = ~no_prof
        out = np.zeros_like(v)
        out[ok] = (v[ok] - v[ok].mean()) / v[ok].std()
        return out

    X = np.column_stack([
        np.ones(len(pr)), no_prof, (pr.gender == "F").to_numpy(), (pr.gender == "M").to_numpy(),
        z(pr.age), z(pr.income.fillna(0)), (tenure - tenure.mean()) / tenure.std(),
    ]).astype(float)
    users = pd.DataFrame(X, columns=CTX_NAMES).assign(person=pr.id.to_numpy())

    tr["offer_id"] = tr.value.apply(lambda v: v.get("offer id", v.get("offer_id")))
    tr["amount"] = tr.value.apply(lambda v: v.get("amount", np.nan))
    rec = tr[tr.event == "offer received"].merge(po[["id", "offer"]], left_on="offer_id", right_on="id")
    txn = tr[tr.event == "transaction"][["person", "time", "amount"]]

    rows = []
    for w in sorted(rec.time.unique()):
        got = rec[rec.time == w].set_index("person").offer
        t = txn[(txn.time >= w) & (txn.time < w + WINDOW_H)]
        spend = t.groupby("person").amount.sum()
        wave = pd.DataFrame({"person": pr.id, "wave": w})
        wave["offer"] = wave.person.map(got).fillna(-1).astype(int)
        wave["spend"] = wave.person.map(spend).fillna(0.0)
        rows.append(wave)
    waves = pd.concat(rows, ignore_index=True)
    waves["buy"] = (waves.spend > 0).astype(int)
    waves = waves.merge(users[["person"]].reset_index(names="uid"), on="person")
    return po, users, waves


# ---------------------------------------------------------------- действия как сочетания критериев

# Критерии и их значения. Первое значение каждого критерия — базовое (в логистике кодируется нулями).
CRITERIA = {
    "type": ["informational", "bogo", "discount"],
    "threshold": ["easy", "hard"],     # порог трат: ≤7 $ / ≥10 $
    "duration": ["short", "long"],     # срок: ≤5 дней / ≥7 дней
    "social": ["no", "yes"],
    "web": ["no", "yes"],
    "mobile": ["no", "yes"],
    "slot": ["morning", "day", "evening", "night"],  # только в симуляторе
}


@dataclass
class Actions:
    """Множество действий: C — индексы значений критериев (A × n_crit), Phi — dummy-признаки (A × K)."""

    criteria: list[str]
    C: np.ndarray
    Phi: np.ndarray
    names: list[str]
    feat_names: list[str]

    @property
    def n(self):
        return len(self.C)

    @staticmethod
    def from_rows(criteria, rows):
        C = np.array(rows, dtype=int)
        cols, fnames = [], []
        for j, c in enumerate(criteria):
            for v in range(1, len(CRITERIA[c])):
                cols.append((C[:, j] == v).astype(float))
                fnames.append(f"{c}={CRITERIA[c][v]}")
        names = [" · ".join(CRITERIA[c][r[j]] for j, c in enumerate(criteria)) for r in rows]
        return Actions(list(criteria), C, np.column_stack(cols), names, fnames)


def portfolio_actions(po) -> Actions:
    """10 реальных офферов как сочетания шести критериев."""
    crit = ["type", "threshold", "duration", "social", "web", "mobile"]
    t = {"informational": 0, "bogo": 1, "discount": 2}
    rows = [[t[r.offer_type], r.hard, r.long, r.social, r.web, r.mobile] for r in po.itertuples()]
    return Actions.from_rows(crit, rows)


def factorial_actions(with_slot=True) -> Actions:
    """Все допустимые сочетания: у информационного оффера нет порога и он короткий."""
    crit = ["type", "threshold", "duration", "social", "web", "mobile"] + (["slot"] if with_slot else [])
    rows = []
    for r in product(*[range(len(CRITERIA[c])) for c in crit]):
        if r[0] == 0 and (r[1] == 1 or r[2] == 1):
            continue
        rows.append(list(r))
    return Actions.from_rows(crit, rows)


# ---------------------------------------------------------------- байесовская логистика (Лаплас)

class LaplaceLogistic:
    """Апостериор весов ≈ N(m, Q⁻¹). Обновление пакетом: MAP методом Ньютона, Q ← кривизна в MAP."""

    def __init__(self, d, prior_sd=1.0, m0=None):
        self.m = np.zeros(d) if m0 is None else np.asarray(m0, float).copy()
        self.Q = np.eye(d) / prior_sd**2

    def update(self, Z, y, iters=8):
        if len(y) == 0:
            return
        m0, Q0, w = self.m, self.Q, self.m.copy()
        for _ in range(iters):
            p = sigmoid(Z @ w)
            grad = Z.T @ (p - y) + Q0 @ (w - m0)
            H = (Z * (p * (1 - p))[:, None]).T @ Z + Q0
            w = w - np.linalg.solve(H, grad)
        p = sigmoid(Z @ w)
        self.m, self.Q = w, (Z * (p * (1 - p))[:, None]).T @ Z + Q0

    def sample(self, n, rng):
        L = np.linalg.cholesky(np.linalg.inv(self.Q))
        return self.m + rng.standard_normal((n, len(self.m))) @ L.T


# ---------------------------------------------------------------- политики
# Интерфейс: choose(X, rng) → индексы действий; update(X, a, y) — пакет наблюдений.

class RandomPolicy:
    label = "Случайная"

    def __init__(self, actions: Actions):
        self.A = actions

    def choose(self, X, rng):
        return rng.integers(self.A.n, size=len(X))

    def update(self, X, a, y):
        pass


class FlatBeta:
    """Плоский бандит: Beta(1,1) на каждое сочетание, без контекста."""

    label = "Beta на сочетание"

    def __init__(self, actions: Actions):
        self.A = actions
        self.alpha = np.ones(actions.n)
        self.beta = np.ones(actions.n)

    def choose(self, X, rng):
        theta = rng.beta(self.alpha, self.beta, size=(len(X), self.A.n))
        return theta.argmax(axis=1)

    def update(self, X, a, y):
        np.add.at(self.alpha, a, y)
        np.add.at(self.beta, a, 1 - y)


class CriterionBeta:
    """Свой Beta-бандит на каждый критерий; сочетание оценивается суммой логитов (Aampe, ступень 1).

    Отклик засчитывается всем значениям критериев отправленного сочетания.
    При аддитивной оценке argmax по сочетаниям = argmax по каждому критерию отдельно
    (если ограничения допустимости не мешают), поэтому это и есть «бандит на критерий».
    """

    label = "Бандит на критерий"

    def __init__(self, actions: Actions):
        self.A = actions
        self.alpha = [np.ones(len(CRITERIA[c])) for c in actions.criteria]
        self.beta = [np.ones(len(CRITERIA[c])) for c in actions.criteria]

    def choose(self, X, rng):
        score = np.zeros((len(X), self.A.n))
        for j in range(len(self.A.criteria)):
            theta = rng.beta(self.alpha[j], self.beta[j], size=(len(X), len(self.alpha[j])))
            score += logit(theta)[:, self.A.C[:, j]]
        return score.argmax(axis=1)

    def update(self, X, a, y):
        for j in range(len(self.A.criteria)):
            v = self.A.C[a, j]
            np.add.at(self.alpha[j], v, y)
            np.add.at(self.beta[j], v, 1 - y)


class LabelLogisticTS:
    """Байесовская логистика по компонентам, Томпсон по весам (Aampe, ступень 2).

    Без контекста: η = w₀ + Σ_k w_k φ_k(a).
    С контекстом: η = w_x·x + Σ_k φ_k(a) (w_k + v_k·x'), x' — контекст без константы.
    """

    def __init__(self, actions: Actions, d_ctx: int, contextual: bool, prior_sd=1.0, b0=0.0):
        self.A, self.ctx = actions, contextual
        self.K = actions.Phi.shape[1]
        self.dx = d_ctx if contextual else 1
        d = self.dx + self.K + (self.K * (d_ctx - 1) if contextual else 0)
        m0 = np.zeros(d)
        m0[0] = b0
        self.model = LaplaceLogistic(d, prior_sd, m0)
        self.label = "Логистика по компонентам" + (" + контекст" if contextual else "")

    def features(self, X, a):
        phi = self.A.Phi[a]
        if not self.ctx:
            return np.column_stack([np.ones(len(a)), phi])
        inter = (phi[:, :, None] * X[:, None, 1:]).reshape(len(a), -1)
        return np.column_stack([X, phi, inter])

    def coef(self, W, X):
        """Коэффициенты при φ(a) для каждого клиента: (n × K)."""
        wk = W[:, self.dx:self.dx + self.K]
        if not self.ctx:
            return wk
        V = W[:, self.dx + self.K:].reshape(len(W), self.K, -1)
        return wk + np.einsum("nkj,nj->nk", V, X[:, 1:])

    def choose(self, X, rng):
        W = self.model.sample(len(X), rng)
        return (self.coef(W, X) @ self.A.Phi.T).argmax(axis=1)

    def update(self, X, a, y):
        self.model.update(self.features(X, a), np.asarray(y, float))

    def mean_effects(self, X):
        """Апостериорное среднее коэффициентов при компонентах для заданных клиентов."""
        return self.coef(self.model.m[None, :].repeat(len(X), 0), X)


class DisjointLogisticTS:
    """Логистика на каждое действие отдельно (аналог disjoint LinTS): η = w_a·x."""

    label = "Логистика на сочетание + контекст"

    def __init__(self, actions: Actions, d_ctx: int, prior_sd=1.0, b0=0.0):
        self.A, self.dx = actions, d_ctx
        m0 = np.zeros(actions.n * d_ctx)
        m0[::d_ctx] = b0
        self.model = LaplaceLogistic(actions.n * d_ctx, prior_sd, m0)

    def features(self, X, a):
        Z = np.zeros((len(a), self.A.n * self.dx))
        for i, (ai, x) in enumerate(zip(a, X)):
            Z[i, ai * self.dx:(ai + 1) * self.dx] = x
        return Z

    def choose(self, X, rng):
        W = self.model.sample(len(X), rng).reshape(len(X), self.A.n, self.dx)
        return np.einsum("nad,nd->na", W, X).argmax(axis=1)

    def update(self, X, a, y):
        self.model.update(self.features(X, a), np.asarray(y, float))


# ---------------------------------------------------------------- офлайн-оценка replay

def replay(policy, X, a_log, y, rng, chunk=1000):
    """Replay (Li et al., 2011) для логов с равномерным выбором из policy.A.n действий.

    События идут в случайном порядке; политика выбирает действие пачкой по `chunk` событий,
    совпавшие с логом события открывают награду и сразу идут в обновление.
    Возвращает награды принятых событий по порядку.
    """
    order = rng.permutation(len(y))
    rewards = []
    for s in range(0, len(order), chunk):
        idx = order[s:s + chunk]
        pick = policy.choose(X[idx], rng)
        hit = pick == a_log[idx]
        acc = idx[hit]
        if len(acc):
            policy.update(X[acc], a_log[acc], y[acc])
            rewards.append(y[acc])
    return np.concatenate(rewards)


# ---------------------------------------------------------------- полусинтетическая среда

class TrueEnv:
    """Истинный отклик: η = w_x·x + sent·(w_s·x) + Σ_k φ_k(a)(w_k + v_k·x') + β_slot(a, x).

    Всё, кроме слота, подогнано MAP-логистикой по реальным волнам (вместе с волнами без оффера).
    Эффект слота задаётся симулятором: свой сдвиг на каждый слот и наклон по возрасту.
    """

    SLOT_BASE = np.array([0.0, 0.10, 0.25, -0.40])     # утро, день, вечер, ночь
    SLOT_AGE = np.array([0.30, 0.0, -0.30, -0.10])      # молодым — вечер, старшим — утро

    def __init__(self, po, users, waves, prior_sd=1.0):
        self.port = portfolio_actions(po)
        X = users[CTX_NAMES].to_numpy()
        self.d = X.shape[1]
        self.K = self.port.Phi.shape[1]
        Xw = X[waves.uid.to_numpy()]
        sent = (waves.offer.to_numpy() >= 0).astype(float)
        phi = np.zeros((len(waves), self.K))
        phi[sent == 1] = self.port.Phi[waves.offer.to_numpy()[sent == 1]]
        Z = self._z(Xw, sent, phi)
        fit = LaplaceLogistic(Z.shape[1], prior_sd)
        fit.update(Z, waves.buy.to_numpy(float), iters=15)
        self.w = fit.m
        self.w_sd = np.sqrt(np.diag(np.linalg.inv(fit.Q)))
        self.feat_names = (CTX_NAMES + [f"sent×{c}" for c in CTX_NAMES] + self.port.feat_names
                           + [f"{f}×{c}" for f in self.port.feat_names for c in CTX_NAMES[1:]])

    def _z(self, X, sent, phi):
        inter = (phi[:, :, None] * X[:, None, 1:]).reshape(len(X), -1)
        return np.column_stack([X, sent[:, None] * X, phi, inter])

    def prob(self, X, actions: Actions, a=None):
        """Матрица вероятностей n × A (или вектор, если задан a); a = -1 — без оффера."""
        d, K = self.d, self.K
        wx, ws = self.w[:d], self.w[d:2 * d]
        wk = self.w[2 * d:2 * d + K]
        V = self.w[2 * d + K:].reshape(K, d - 1)
        base = X @ wx
        coef = wk[None, :] + X[:, 1:] @ V.T                         # n × K
        phi = actions.Phi[:, :K]                                    # первые K — критерии портфеля
        eta = base[:, None] + (X @ ws)[:, None] + coef @ phi.T      # n × A
        if "slot" in actions.criteria:
            s = actions.C[:, actions.criteria.index("slot")]
            age = X[:, CTX_NAMES.index("age")]
            eta = eta + self.SLOT_BASE[s][None, :] + age[:, None] * self.SLOT_AGE[s][None, :]
        p = sigmoid(eta)
        if a is None:
            return p
        return p[np.arange(len(X)), a]

    def prob_none(self, X):
        return sigmoid(X @ self.w[:self.d])


def simulate(policy, env: TrueEnv, actions: Actions, X_pool, days, per_day, rng):
    """Онлайн-прогон: каждый день per_day случайных клиентов, обновление пакетом дня.

    Возвращает по дням: ожидаемое сожаление, ожидаемую покупку, ожидаемый прирост к «без оффера».
    """
    regret, value, uplift = [], [], []
    for _ in range(days):
        X = X_pool[rng.integers(len(X_pool), size=per_day)]
        P = env.prob(X, actions)
        a = policy.choose(X, rng)
        p = P[np.arange(len(X)), a]
        y = (rng.random(len(X)) < p).astype(float)
        policy.update(X, a, y)
        regret.append((P.max(axis=1) - p).sum())
        value.append(p.mean())
        uplift.append((p - env.prob_none(X)).mean())
    return np.array(regret), np.array(value), np.array(uplift)
