import numpy as np
import pandas as pd
import cvxpy as cp


def solve_problem(problem):
    for solver in ("ECOS", "OSQP"):
        try:
            problem.solve(solver=solver)
        except Exception:
            continue
        if problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            return
    problem.solve()


def add_hourly_constant_constraints(P, time_index):
    time_index = pd.DatetimeIndex(time_index)
    hours = time_index.floor("h")
    constraints = []
    n = len(time_index)
    start = 0
    while start < n:
        hour = hours[start]
        end = start + 1
        while end < n and hours[end] == hour:
            constraints.append(P[end] == P[start])
            end += 1
        start = end
    return constraints


def compute_cost(prices, power_kw, dt_hours):
    return float(np.sum(prices * power_kw) * dt_hours)


def check_hourly_constant(power_kw, time_index, tol=1e-4):
    time_index = pd.DatetimeIndex(time_index)
    hours = time_index.floor("h")
    n = len(power_kw)
    start = 0
    while start < n:
        hour = hours[start]
        end = start + 1
        while end < n and hours[end] == hour:
            if abs(power_kw[end] - power_kw[start]) > tol:
                return False
            end += 1
        start = end
    return True


def check_ev_energy(power_kw, dt_hours, eta, e_need, tol=1e-6):
    grid_energy = float(np.sum(power_kw) * dt_hours)
    target = e_need / eta
    return abs(grid_energy - target) <= tol


def check_tank_bounds(temp_c, tmin, tmax, tol=1e-6):
    temp_c = np.asarray(temp_c, dtype=float)
    below = temp_c < (tmin - tol)
    above = temp_c > (tmax + tol)
    return int(np.sum(below) + np.sum(above))


def optimize_ev(
    prices,
    time_index,
    pmax,
    eta,
    e_need,
    dt_hours=0.25,
    hourly_control=False,
):
    prices = np.asarray(prices, dtype=float)
    n = prices.size
    if n == 0:
        return {"status": "empty", "cost": np.nan, "P": None, "charged_kwh": np.nan}

    P = cp.Variable(n)
    e_grid = e_need / eta
    constraints = [
        P >= 0.0,
        P <= pmax,
        cp.sum(P) * dt_hours == e_grid,
    ]
    if hourly_control:
        constraints += add_hourly_constant_constraints(P, time_index)

    cost_expr = cp.sum(cp.multiply(prices, P)) * dt_hours
    problem = cp.Problem(cp.Minimize(cost_expr), constraints)
    solve_problem(problem)

    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        return {"status": problem.status, "cost": np.nan, "P": None, "charged_kwh": np.nan}

    P_val = np.asarray(P.value, dtype=float)
    cost = compute_cost(prices, P_val, dt_hours)
    charged_kwh = float(np.sum(P_val) * dt_hours * eta)
    return {
        "status": problem.status,
        "cost": cost,
        "P": P_val,
        "charged_kwh": charged_kwh,
    }


def optimize_tank(
    prices,
    time_index,
    draw_kwh,
    params,
    dt_hours=0.25,
    hourly_control=False,
):
    prices = np.asarray(prices, dtype=float)
    draw_kwh = np.asarray(draw_kwh, dtype=float)
    n = prices.size
    if draw_kwh.size != n:
        raise ValueError("draw_kwh length must match price length")
    if n == 0:
        return {
            "status": "empty",
            "cost": np.nan,
            "P": None,
            "T": None,
            "min_temp": np.nan,
            "max_temp": np.nan,
            "violations": np.nan,
        }

    pmax = params["pmax"]
    eta = params["eta"]
    tmin = params["tmin"]
    tmax = params["tmax"]
    t0 = params["t0"]
    tamb = params["tamb"]
    k_loss = params["k_loss"]
    c_kwh_per_c = params["c_kwh_per_c"]

    Qdraw_kw = draw_kwh / dt_hours

    P = cp.Variable(n)
    T = cp.Variable(n + 1)

    constraints = [
        T[0] == t0,
        P >= 0.0,
        P <= pmax,
        T >= tmin,
        T <= tmax,
    ]
    for t in range(n):
        loss_kw = k_loss * (T[t] - tamb)
        constraints.append(
            T[t + 1]
            == T[t]
            + (dt_hours / c_kwh_per_c) * (eta * P[t] - loss_kw - Qdraw_kw[t])
        )
    if hourly_control:
        constraints += add_hourly_constant_constraints(P, time_index)

    cost_expr = cp.sum(cp.multiply(prices, P)) * dt_hours
    problem = cp.Problem(cp.Minimize(cost_expr), constraints)
    solve_problem(problem)

    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        return {
            "status": problem.status,
            "cost": np.nan,
            "P": None,
            "T": None,
            "min_temp": np.nan,
            "max_temp": np.nan,
            "violations": np.nan,
        }

    P_val = np.asarray(P.value, dtype=float)
    T_val = np.asarray(T.value, dtype=float)
    cost = compute_cost(prices, P_val, dt_hours)
    min_temp = float(np.min(T_val))
    max_temp = float(np.max(T_val))
    violations = check_tank_bounds(T_val, tmin, tmax, tol=1e-4)
    return {
        "status": problem.status,
        "cost": cost,
        "P": P_val,
        "T": T_val,
        "min_temp": min_temp,
        "max_temp": max_temp,
        "violations": violations,
    }
