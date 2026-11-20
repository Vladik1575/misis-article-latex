# Эксперименты статьи: бандиты по компонентам оффера (Starbucks)

- `sblab.py` — данные, политики (бандит на сочетание, бандит на критерий с суммой логитов, логистика по компонентам с контекстом и без, логистика на сочетание), replay, полусинтетическая среда.
- `make_notebook.py` → `starbucks_bandits.ipynb` — все эксперименты; пишет `results.json` и графики в `../figures/`.
- `data/` — не в git. Скачать: `kaggle datasets download blacktile/starbucks-app-customer-reward-program-data -p data --unzip`.

Запуск (~1 мин на ноутбуке):

    python3 -m venv .venv && .venv/bin/pip install numpy pandas matplotlib scipy nbformat nbclient ipykernel
    .venv/bin/python make_notebook.py   # собрать ноутбук, затем открыть и выполнить его
