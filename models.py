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


def optimize_tank_softmin(
    prices,
    time_index,
    draw_kwh,
    params,
    dt_hours=0.25,
    hourly_control=False,
    lambda_slack=100.0,
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
            "Temp": None,
            "slack_sum": np.nan,
            "slack_max": np.nan,
            "min_temp": np.nan,
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
    Temp = cp.Variable(n + 1)
    slack = cp.Variable(n + 1, nonneg=True)

    constraints = [
        Temp[0] == t0,
        P >= 0.0,
        P <= pmax,
        Temp <= tmax,
        Temp >= tmin - slack,
    ]
    for t in range(n):
        loss_kw = k_loss * (Temp[t] - tamb)
        constraints.append(
            Temp[t + 1]
            == Temp[t]
            + (dt_hours / c_kwh_per_c) * (eta * P[t] - loss_kw - Qdraw_kw[t])
        )
    if hourly_control:
        constraints += add_hourly_constant_constraints(P, time_index)

    cost_energy = cp.sum(cp.multiply(prices, P)) * dt_hours
    cost = cost_energy + lambda_slack * cp.sum(slack) * dt_hours
    problem = cp.Problem(cp.Minimize(cost), constraints)
    solve_problem(problem)

    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        return {
            "status": problem.status,
            "cost": np.nan,
            "P": None,
            "Temp": None,
            "slack_sum": np.nan,
            "slack_max": np.nan,
            "min_temp": np.nan,
        }

    P_val = np.asarray(P.value, dtype=float)
    Temp_val = np.asarray(Temp.value, dtype=float)
    slack_val = np.asarray(slack.value, dtype=float)
    slack_sum = float(np.sum(slack_val) * dt_hours)
    slack_max = float(np.max(slack_val))
    min_temp = float(np.min(Temp_val))
    cost_val = float(problem.value) if problem.value is not None else np.nan
    return {
        "status": problem.status,
        "cost": cost_val,
        "P": P_val,
        "Temp": Temp_val,
        "slack_sum": slack_sum,
        "slack_max": slack_max,
        "min_temp": min_temp,
    }


def baseline_ev_charge_now(prices_eur_per_kwh, dt_hours, pmax_kw, eta, e_need_kwh):
    prices = np.asarray(prices_eur_per_kwh, dtype=float)
    n = prices.size
    P = np.zeros(n, dtype=float)
    if n == 0:
        return {
            "status": "empty",
            "cost": np.nan,
            "P": P,
            "charged_grid_kwh": np.nan,
        }

    e_grid = e_need_kwh / eta
    remaining = float(e_grid)
    max_step_kwh = pmax_kw * dt_hours
    tol = 1e-6

    for t in range(n):
        if remaining <= tol:
            break
        step_kwh = min(max_step_kwh, remaining)
        P[t] = step_kwh / dt_hours
        remaining -= step_kwh

    if remaining > tol:
        return {"status": "infeasible", "cost": np.nan, "P": P, "charged_grid_kwh": e_grid - remaining}

    cost = compute_cost(prices, P, dt_hours)
    charged_grid_kwh = float(np.sum(P) * dt_hours)
    return {"status": "simulated", "cost": cost, "P": P, "charged_grid_kwh": charged_grid_kwh}


def baseline_ev_fixed_window(
    prices_eur_per_kwh,
    time_index,
    dt_hours,
    pmax_kw,
    eta,
    e_need_kwh,
    start_h=0,
    end_h=6,
):
    prices = np.asarray(prices_eur_per_kwh, dtype=float)
    times = pd.DatetimeIndex(time_index)
    n = prices.size
    P = np.zeros(n, dtype=float)
    if n == 0:
        return {
            "status": "empty",
            "cost": np.nan,
            "P": P,
            "charged_grid_kwh": np.nan,
            "allowed_count": 0,
        }

    allowed = (times.hour >= start_h) & (times.hour < end_h)
    allowed_count = int(np.sum(allowed))
    if allowed_count == 0:
        return {
            "status": "infeasible",
            "cost": np.nan,
            "P": P,
            "charged_grid_kwh": 0.0,
            "allowed_count": allowed_count,
        }

    e_grid = e_need_kwh / eta
    e_max = allowed_count * pmax_kw * dt_hours
    tol = 1e-6
    if e_max < e_grid - tol:
        return {
            "status": "infeasible",
            "cost": np.nan,
            "P": P,
            "charged_grid_kwh": e_max,
            "allowed_count": allowed_count,
        }

    p_allowed = e_grid / (allowed_count * dt_hours)
    if p_allowed > pmax_kw + tol:
        return {
            "status": "infeasible",
            "cost": np.nan,
            "P": P,
            "charged_grid_kwh": e_grid,
            "allowed_count": allowed_count,
        }

    P[allowed] = min(p_allowed, pmax_kw)
    cost = compute_cost(prices, P, dt_hours)
    charged_grid_kwh = float(np.sum(P) * dt_hours)
    return {
        "status": "simulated",
        "cost": cost,
        "P": P,
        "charged_grid_kwh": charged_grid_kwh,
        "allowed_count": allowed_count,
    }


def _tank_step(
    T_curr,
    dt_hours,
    eta,
    p_kw,
    tamb,
    c_kwh_per_c,
    k_kw_per_c,
    qdraw_kw,
):
    loss_kw = k_kw_per_c * (T_curr - tamb)
    return T_curr + (dt_hours / c_kwh_per_c) * (eta * p_kw - loss_kw - qdraw_kw)


def _tank_power_for_target(
    T_curr,
    T_target,
    dt_hours,
    eta,
    tamb,
    c_kwh_per_c,
    k_kw_per_c,
    qdraw_kw,
):
    loss_kw = k_kw_per_c * (T_curr - tamb)
    needed_kw = ((T_target - T_curr) * c_kwh_per_c / dt_hours) + loss_kw + qdraw_kw
    return needed_kw / eta


def baseline_tank_thermostat(
    prices_eur_per_kwh,
    dt_hours,
    pmax_kw,
    eta,
    T0,
    Tmin,
    Tmax,
    Tamb,
    C_kwh_per_C,
    k_kw_per_C,
    Qdraw_kw,
):
    prices = np.asarray(prices_eur_per_kwh, dtype=float)
    qdraw = np.asarray(Qdraw_kw, dtype=float)
    n = prices.size
    P = np.zeros(n, dtype=float)
    T = np.zeros(n + 1, dtype=float)
    if qdraw.size != n:
        raise ValueError("Qdraw_kw length must match price length")
    if n == 0:
        return {
            "status": "empty",
            "cost": np.nan,
            "P": P,
            "T": T,
            "min_temp": np.nan,
            "max_temp": np.nan,
            "violations": np.nan,
        }

    T[0] = T0
    for t in range(n):
        t_no_heat = _tank_step(
            T[t], dt_hours, eta, 0.0, Tamb, C_kwh_per_C, k_kw_per_C, qdraw[t]
        )
        if t_no_heat < Tmin:
            p_needed = _tank_power_for_target(
                T[t],
                Tmin,
                dt_hours,
                eta,
                Tamb,
                C_kwh_per_C,
                k_kw_per_C,
                qdraw[t],
            )
            p_kw = np.clip(p_needed, 0.0, pmax_kw)
        else:
            p_kw = 0.0

        t_next = _tank_step(
            T[t], dt_hours, eta, p_kw, Tamb, C_kwh_per_C, k_kw_per_C, qdraw[t]
        )
        if t_next > Tmax:
            p_cap = _tank_power_for_target(
                T[t],
                Tmax,
                dt_hours,
                eta,
                Tamb,
                C_kwh_per_C,
                k_kw_per_C,
                qdraw[t],
            )
            p_kw = np.clip(min(p_kw, p_cap), 0.0, pmax_kw)
            t_next = _tank_step(
                T[t], dt_hours, eta, p_kw, Tamb, C_kwh_per_C, k_kw_per_C, qdraw[t]
            )

        P[t] = p_kw
        T[t + 1] = t_next

    cost = compute_cost(prices, P, dt_hours)
    min_temp = float(np.min(T))
    max_temp = float(np.max(T))
    violations = check_tank_bounds(T, Tmin, Tmax, tol=1e-3)
    return {
        "status": "simulated",
        "cost": cost,
        "P": P,
        "T": T,
        "min_temp": min_temp,
        "max_temp": max_temp,
        "violations": violations,
    }


def baseline_tank_night_only(
    prices_eur_per_kwh,
    time_index,
    dt_hours,
    pmax_kw,
    eta,
    T0,
    Tmin,
    Tmax,
    Tamb,
    C_kwh_per_C,
    k_kw_per_C,
    Qdraw_kw,
    start_h=0,
    end_h=6,
):
    prices = np.asarray(prices_eur_per_kwh, dtype=float)
    times = pd.DatetimeIndex(time_index)
    qdraw = np.asarray(Qdraw_kw, dtype=float)
    n = prices.size
    P = np.zeros(n, dtype=float)
    T = np.zeros(n + 1, dtype=float)
    if qdraw.size != n:
        raise ValueError("Qdraw_kw length must match price length")
    if n == 0:
        return {
            "status": "empty",
            "cost": np.nan,
            "P": P,
            "T": T,
            "min_temp": np.nan,
            "max_temp": np.nan,
            "violations": np.nan,
        }

    T[0] = T0
    night_mask = (times.hour >= start_h) & (times.hour < end_h)
    for t in range(n):
        t_no_heat = _tank_step(
            T[t], dt_hours, eta, 0.0, Tamb, C_kwh_per_C, k_kw_per_C, qdraw[t]
        )

        if night_mask[t]:
            target = Tmax
            p_needed = _tank_power_for_target(
                T[t],
                target,
                dt_hours,
                eta,
                Tamb,
                C_kwh_per_C,
                k_kw_per_C,
                qdraw[t],
            )
            p_kw = np.clip(p_needed, 0.0, pmax_kw)
        elif t_no_heat < Tmin:
            p_needed = _tank_power_for_target(
                T[t],
                Tmin,
                dt_hours,
                eta,
                Tamb,
                C_kwh_per_C,
                k_kw_per_C,
                qdraw[t],
            )
            p_kw = np.clip(p_needed, 0.0, pmax_kw)
        else:
            p_kw = 0.0

        t_next = _tank_step(
            T[t], dt_hours, eta, p_kw, Tamb, C_kwh_per_C, k_kw_per_C, qdraw[t]
        )
        if t_next > Tmax:
            p_cap = _tank_power_for_target(
                T[t],
                Tmax,
                dt_hours,
                eta,
                Tamb,
                C_kwh_per_C,
                k_kw_per_C,
                qdraw[t],
            )
            p_kw = np.clip(min(p_kw, p_cap), 0.0, pmax_kw)
            t_next = _tank_step(
                T[t], dt_hours, eta, p_kw, Tamb, C_kwh_per_C, k_kw_per_C, qdraw[t]
            )

        P[t] = p_kw
        T[t + 1] = t_next

    cost = compute_cost(prices, P, dt_hours)
    min_temp = float(np.min(T))
    max_temp = float(np.max(T))
    violations = check_tank_bounds(T, Tmin, Tmax, tol=1e-3)
    return {
        "status": "simulated",
        "cost": cost,
        "P": P,
        "T": T,
        "min_temp": min_temp,
        "max_temp": max_temp,
        "violations": violations,
    }
