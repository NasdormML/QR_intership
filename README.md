## Задача по Quantitative Research
---
1) Загрузите котировки акций-голубых фишек российского рынка за последние полгода.
2) Скорректируйте котировки на сплиты.
3) Постройте Mean-Variance Portfolio из акций, загруженных ранее, и оцените его динамику.

**Условия к файлу:**
Результат должен быть представлен в Google colab, Jupyter notebook или Python скрипте.

---

**Инструменты**
- Python 3.10+  
- `pandas`, `numpy`  
- ISS Statistics API  
- `scikit-learn` (LedoitWolf)  
- `cvxpy`  
- `matplotlib`

**Решение:**
- Выгружаем список голубых фишек MOEXBC через ISS statistics API их "сырые" цены и проверяем, были ли сплиты в нашем диапозоне.
- Для постройки MVP метод **LedoitWolf** библиотеки sklearn.

**Анализ ковариации и доходностей:**
```python
# returns — DataFrame с ежедневными доходностями акций
# tickers — список или Index тикеров
lw = LedoitWolf().fit(returns)
Sigma = pd.DataFrame(lw.covariance_, index=returns.columns, columns=returns.columns) * 252
mu = returns.mean() * 252
```
**Формулировка и решение MV-задачи:**
```python
# 6. MV оптимизация
tickers = returns.columns.tolist()
n = len(tickers)
w = cp.Variable(n)
lam = 1

objective = cp.Maximize(mu.values @ w - lam * cp.quad_form(w, Sigma.values))
constraints = [cp.sum(w) == 1, w >= 0.05, w <= 0.25]
prob = cp.Problem(objective, constraints)

# 7. Расчет показателей портфеля
portf_ret = returns.dot(w_opt)
cum_ret = (1 + portf_ret).prod() - 1
n_years = len(returns) / 252
annualized_ret = (1 + cum_ret)**(1/n_years) - 1
annualized_vol = portf_ret.std() * np.sqrt(252)
sharpe_ratio = (annualized_ret - RISK_FREE_RATE) / annualized_vol
```

Ниже показан график поведения портфеля и его метрики за 6 месяцев.
---
![image](https://github.com/user-attachments/assets/7860e444-370d-400a-a478-dd61010aa6a5)

**Cumulative return:** 21.89%
**Annualized Return (CAGR):** 56.10%
**Annualized Volatility:** 34.43%
**Sharpe Ratio:** 1.48
