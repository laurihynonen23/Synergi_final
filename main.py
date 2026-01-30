import argparse
import os

import numpy as np
import pandas as pd

from io_utils import read_prices, make_hourly_price_repeated
from models import (
    optimize_ev,
    optimize_tank,
    optimize_tank_softmin,
    check_hourly_constant,
    check_ev_energy,
    check_tank_bounds,
    baseline_ev_charge_now,
    baseline_ev_fixed_window,
    baseline_tank_thermostat,
    baseline_tank_night_only,
)
from scenarios import build_ev_scenarios, build_tank_scenarios


DT_HOURS = 0.25
EPS_EUR = 1e-6
OPTIMAL_STATUSES = {"optimal", "optimal_inaccurate"}
FEASIBLE_STATUSES = OPTIMAL_STATUSES | {"simulated"}


def _compute_delta(cost_60, cost_15, scenario_id="", method="OPT"):
    if np.isnan(cost_60) or np.isnan(cost_15):
        return np.nan, np.nan
    delta = cost_60 - cost_15
    if np.isfinite(delta) and abs(delta) < EPS_EUR:
        return 0.0, 0.0
    if np.isfinite(delta) and delta < -1e-4 and method == "OPT":
        print(f"Warning: unexpected negative delta {delta:.6f} for {scenario_id}")
    if abs(cost_60) < 1e-9:
        return delta, np.nan
    delta_pct = (delta / cost_60) * 100.0
    if delta == 0.0:
        delta_pct = 0.0
    return delta, delta_pct


def _is_optimal(status):
    return status in OPTIMAL_STATUSES


def _is_feasible(status):
    return status in FEASIBLE_STATUSES


def _warn_if_not_optimal(scenario_id, label, status):
    if not _is_optimal(status):
        print(f"Warning: {scenario_id} {label} status={status}")


def _row_status(status_15, status_60):
    if _is_feasible(status_15) and _is_feasible(status_60):
        return "ok"
    return "infeasible"


def validate_daily_draw(time_index, draw_kwh, expected_kwh, scenario_id, tol=1e-3):
    series = pd.Series(draw_kwh, index=pd.DatetimeIndex(time_index))
    daily = series.groupby(series.index.normalize()).sum()
    if not np.allclose(daily.values, expected_kwh, atol=tol):
        raise ValueError(
            f"{scenario_id} daily draw mismatch: {daily.values} expected {expected_kwh}"
        )


def print_profile_summary(ev_scenarios, tank_scenarios):
    print("Profile summary:")
    if ev_scenarios:
        home_window = ev_scenarios[0].get("home_window", "")
        days = sorted({sc["start_time"].normalize() for sc in ev_scenarios})
        combos = sorted({(sc["pmax"], sc["e_need"]) for sc in ev_scenarios})
        print(f"EV window: {home_window} (days={len(days)})")
        for pmax, e_need in combos:
            print(f"EV: Pmax={pmax:g} kW, E_need={e_need:g} kWh")
    if tank_scenarios:
        print("Tank profiles:")
        for sc in tank_scenarios:
            events = sc.get("events", [])
            daily_kwh = sc.get("daily_hotwater_kwh", np.nan)
            print(f"{sc['scenario_id']}: daily_kwh={daily_kwh}, events={events}")


def run_all(time_index, p15, p60, ev_scenarios, tank_scenarios, tank_params):
    results = []
    plot_data = {"ev": None, "tank": None}
    best_ev = None
    best_ev_abs = -1.0
    tank_target = "TANK_family4_6showers"
    tank_fallback = None

    for sc in ev_scenarios:
        start_idx = sc["start_idx"]
        end_idx = sc["end_idx"]
        times = time_index[start_idx:end_idx]
        prices_15 = p15[start_idx:end_idx]
        prices_60 = p60[start_idx:end_idx]

        res_15 = optimize_ev(
            prices_15,
            times,
            pmax=sc["pmax"],
            eta=sc["eta"],
            e_need=sc["e_need"],
            dt_hours=DT_HOURS,
            hourly_control=False,
        )
        res_60 = optimize_ev(
            prices_60,
            times,
            pmax=sc["pmax"],
            eta=sc["eta"],
            e_need=sc["e_need"],
            dt_hours=DT_HOURS,
            hourly_control=True,
        )
        res_ctrl = optimize_ev(
            prices_15,
            times,
            pmax=sc["pmax"],
            eta=sc["eta"],
            e_need=sc["e_need"],
            dt_hours=DT_HOURS,
            hourly_control=True,
        )
        res_price = optimize_ev(
            prices_60,
            times,
            pmax=sc["pmax"],
            eta=sc["eta"],
            e_need=sc["e_need"],
            dt_hours=DT_HOURS,
            hourly_control=False,
        )

        for label, res, hourly in (
            ("15min", res_15, False),
            ("60min", res_60, True),
            ("ctrl_only", res_ctrl, True),
            ("price_only", res_price, False),
        ):
            _warn_if_not_optimal(sc["scenario_id"], label, res["status"])
            if not _is_optimal(res["status"]) or res["P"] is None:
                continue
            if not check_ev_energy(res["P"], DT_HOURS, sc["eta"], sc["e_need"], tol=1e-3):
                raise ValueError(f"EV energy constraint failed: {sc['scenario_id']}")
            if hourly and not check_hourly_constant(res["P"], times, tol=1e-4):
                raise ValueError(f"EV hourly constraint failed: {sc['scenario_id']}")

        delta_eur, delta_pct = _compute_delta(
            res_60["cost"],
            res_15["cost"],
            scenario_id=sc["scenario_id"],
            method="OPT",
        )
        results.append(
            {
                "method": "OPT",
                "device": sc["device"],
                "scenario_id": sc["scenario_id"],
                "home_window": sc.get("home_window", ""),
                "pmax": sc["pmax"],
                "e_need": sc["e_need"],
                "cost_15min": res_15["cost"],
                "cost_60min": res_60["cost"],
                "delta_eur": delta_eur,
                "delta_pct": delta_pct,
                "cost_ctrl_only": res_ctrl["cost"],
                "cost_price_only": res_price["cost"],
                "status_15min": res_15["status"],
                "status_60min": res_60["status"],
                "status_ctrl_only": res_ctrl["status"],
                "status_price_only": res_price["status"],
                "row_status": _row_status(res_15["status"], res_60["status"]),
                "min_temp_15": np.nan,
                "min_temp_60": np.nan,
                "violations_15": np.nan,
                "violations_60": np.nan,
                "daily_hotwater_kwh": np.nan,
                "events": "",
            }
        )

        if (
            _is_optimal(res_15["status"])
            and _is_optimal(res_60["status"])
            and res_15["P"] is not None
            and res_60["P"] is not None
            and not np.isnan(delta_eur)
        ):
            abs_delta = abs(delta_eur)
            if abs_delta > best_ev_abs:
                best_ev_abs = abs_delta
                best_ev = {
                    "time": pd.DatetimeIndex(times),
                    "P_15": res_15["P"],
                    "P_60": res_60["P"],
                    "scenario_id": sc["scenario_id"],
                }

    if best_ev is not None:
        plot_data["ev"] = best_ev

    for sc in tank_scenarios:
        daily_kwh = sc.get("daily_hotwater_kwh")
        if daily_kwh is not None and not np.isnan(daily_kwh):
            validate_daily_draw(time_index, sc["draw_kwh"], daily_kwh, sc["scenario_id"])

        prices_15 = p15
        prices_60 = p60
        draw_kwh = sc["draw_kwh"]

        res_15 = optimize_tank(
            prices_15,
            time_index,
            draw_kwh,
            tank_params,
            dt_hours=DT_HOURS,
            hourly_control=False,
        )
        res_60 = optimize_tank(
            prices_60,
            time_index,
            draw_kwh,
            tank_params,
            dt_hours=DT_HOURS,
            hourly_control=True,
        )
        res_ctrl = optimize_tank(
            prices_15,
            time_index,
            draw_kwh,
            tank_params,
            dt_hours=DT_HOURS,
            hourly_control=True,
        )
        res_price = optimize_tank(
            prices_60,
            time_index,
            draw_kwh,
            tank_params,
            dt_hours=DT_HOURS,
            hourly_control=False,
        )

        for label, res, hourly in (
            ("15min", res_15, False),
            ("60min", res_60, True),
            ("ctrl_only", res_ctrl, True),
            ("price_only", res_price, False),
        ):
            _warn_if_not_optimal(sc["scenario_id"], label, res["status"])
            if not _is_optimal(res["status"]) or res["T"] is None:
                continue
            violations = check_tank_bounds(
                res["T"], tank_params["tmin"], tank_params["tmax"], tol=1e-3
            )
            if violations > 0:
                raise ValueError(f"Tank bounds failed: {sc['scenario_id']}")
            if hourly and not check_hourly_constant(res["P"], time_index, tol=1e-4):
                raise ValueError(f"Tank hourly constraint failed: {sc['scenario_id']}")

        delta_eur, delta_pct = _compute_delta(
            res_60["cost"],
            res_15["cost"],
            scenario_id=sc["scenario_id"],
            method="OPT",
        )
        results.append(
            {
                "method": "OPT",
                "device": sc["device"],
                "scenario_id": sc["scenario_id"],
                "cost_15min": res_15["cost"],
                "cost_60min": res_60["cost"],
                "delta_eur": delta_eur,
                "delta_pct": delta_pct,
                "cost_ctrl_only": res_ctrl["cost"],
                "cost_price_only": res_price["cost"],
                "status_15min": res_15["status"],
                "status_60min": res_60["status"],
                "status_ctrl_only": res_ctrl["status"],
                "status_price_only": res_price["status"],
                "row_status": _row_status(res_15["status"], res_60["status"]),
                "min_temp_15": res_15["min_temp"],
                "min_temp_60": res_60["min_temp"],
                "violations_15": res_15["violations"],
                "violations_60": res_60["violations"],
                "daily_hotwater_kwh": sc.get("daily_hotwater_kwh", np.nan),
                "events": sc.get("events", ""),
            }
        )

        if (
            tank_fallback is None
            and _is_optimal(res_15["status"])
            and _is_optimal(res_60["status"])
            and res_15["T"] is not None
            and res_60["T"] is not None
        ):
            tank_fallback = {
                "time": pd.DatetimeIndex(time_index),
                "T_15": res_15["T"],
                "T_60": res_60["T"],
                "scenario_id": sc["scenario_id"],
            }
        if (
            sc["scenario_id"] == tank_target
            and _is_optimal(res_15["status"])
            and _is_optimal(res_60["status"])
            and res_15["T"] is not None
            and res_60["T"] is not None
        ):
            plot_data["tank"] = {
                "time": pd.DatetimeIndex(time_index),
                "T_15": res_15["T"],
                "T_60": res_60["T"],
                "scenario_id": sc["scenario_id"],
            }

    if plot_data["tank"] is None and tank_fallback is not None:
        plot_data["tank"] = tank_fallback

    return results, plot_data


def run_baselines(time_index, p15, p60, ev_scenarios, tank_scenarios, tank_params):
    results = []

    for sc in ev_scenarios:
        start_idx = sc["start_idx"]
        end_idx = sc["end_idx"]
        times = pd.DatetimeIndex(time_index[start_idx:end_idx])
        prices_15 = p15[start_idx:end_idx]
        prices_60 = p60[start_idx:end_idx]

        ev_baselines = [
            (
                "BASE_CHARGE_NOW",
                lambda prices: baseline_ev_charge_now(
                    prices, DT_HOURS, sc["pmax"], sc["eta"], sc["e_need"]
                ),
            ),
            (
                "BASE_FIXED_00_06",
                lambda prices: baseline_ev_fixed_window(
                    prices,
                    times,
                    DT_HOURS,
                    sc["pmax"],
                    sc["eta"],
                    sc["e_need"],
                    start_h=0,
                    end_h=6,
                ),
            ),
        ]

        for method, fn in ev_baselines:
            res_15 = fn(prices_15)
            res_60 = fn(prices_60)

            for res in (res_15, res_60):
                if res.get("status") != "simulated" or res.get("P") is None:
                    continue
                if not check_ev_energy(
                    res["P"], DT_HOURS, sc["eta"], sc["e_need"], tol=1e-3
                ):
                    raise ValueError(f"EV baseline energy failed: {sc['scenario_id']} {method}")

            delta_eur, delta_pct = _compute_delta(
                res_60["cost"],
                res_15["cost"],
                scenario_id=f"{sc['scenario_id']}[{method}]",
                method=method,
            )
            results.append(
                {
                    "method": method,
                    "device": sc["device"],
                    "scenario_id": sc["scenario_id"],
                    "home_window": sc.get("home_window", ""),
                    "pmax": sc["pmax"],
                    "e_need": sc["e_need"],
                    "cost_15min": res_15["cost"],
                    "cost_60min": res_60["cost"],
                    "delta_eur": delta_eur,
                    "delta_pct": delta_pct,
                    "cost_ctrl_only": np.nan,
                    "cost_price_only": np.nan,
                    "status_15min": res_15.get("status", ""),
                    "status_60min": res_60.get("status", ""),
                    "status_ctrl_only": "",
                    "status_price_only": "",
                    "row_status": _row_status(res_15.get("status", ""), res_60.get("status", "")),
                    "min_temp_15": np.nan,
                    "min_temp_60": np.nan,
                    "violations_15": np.nan,
                    "violations_60": np.nan,
                    "daily_hotwater_kwh": np.nan,
                    "events": "",
                }
            )

    qdraw_kw = None
    for sc in tank_scenarios:
        daily_kwh = sc.get("daily_hotwater_kwh")
        if daily_kwh is not None and not np.isnan(daily_kwh):
            validate_daily_draw(time_index, sc["draw_kwh"], daily_kwh, sc["scenario_id"])

        draw_kwh = sc["draw_kwh"]
        qdraw_kw = draw_kwh / DT_HOURS

        tank_baselines = [
            (
                "BASE_THERMOSTAT",
                lambda prices: baseline_tank_thermostat(
                    prices,
                    DT_HOURS,
                    tank_params["pmax"],
                    tank_params["eta"],
                    tank_params["t0"],
                    tank_params["tmin"],
                    tank_params["tmax"],
                    tank_params["tamb"],
                    tank_params["c_kwh_per_c"],
                    tank_params["k_loss"],
                    qdraw_kw,
                ),
            ),
            (
                "BASE_NIGHT_00_06",
                lambda prices: baseline_tank_night_only(
                    prices,
                    time_index,
                    DT_HOURS,
                    tank_params["pmax"],
                    tank_params["eta"],
                    tank_params["t0"],
                    tank_params["tmin"],
                    tank_params["tmax"],
                    tank_params["tamb"],
                    tank_params["c_kwh_per_c"],
                    tank_params["k_loss"],
                    qdraw_kw,
                    start_h=0,
                    end_h=6,
                ),
            ),
        ]

        for method, fn in tank_baselines:
            res_15 = fn(p15)
            res_60 = fn(p60)
            delta_eur, delta_pct = _compute_delta(
                res_60["cost"],
                res_15["cost"],
                scenario_id=f"{sc['scenario_id']}[{method}]",
                method=method,
            )
            results.append(
                {
                    "method": method,
                    "device": sc["device"],
                    "scenario_id": sc["scenario_id"],
                    "cost_15min": res_15["cost"],
                    "cost_60min": res_60["cost"],
                    "delta_eur": delta_eur,
                    "delta_pct": delta_pct,
                    "cost_ctrl_only": np.nan,
                    "cost_price_only": np.nan,
                    "status_15min": res_15.get("status", ""),
                    "status_60min": res_60.get("status", ""),
                    "status_ctrl_only": "",
                    "status_price_only": "",
                    "row_status": _row_status(res_15.get("status", ""), res_60.get("status", "")),
                    "min_temp_15": res_15.get("min_temp", np.nan),
                    "min_temp_60": res_60.get("min_temp", np.nan),
                    "violations_15": res_15.get("violations", np.nan),
                    "violations_60": res_60.get("violations", np.nan),
                    "daily_hotwater_kwh": sc.get("daily_hotwater_kwh", np.nan),
                    "events": sc.get("events", ""),
                }
            )

    return results


def plot_tank_temp_soft(plot_data, out_dir, lambda_slack):
    if plot_data is None:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = pd.DatetimeIndex(plot_data["time"])
    temp_times = times.append(pd.DatetimeIndex([times[-1] + pd.Timedelta(minutes=15)]))

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(temp_times, plot_data["T_15"], label="softmin 15-min control")
    ax.plot(temp_times, plot_data["T_60"], label="softmin 60-min control")
    ax.set_ylabel("C")
    ax.set_title(f"Tank soft Tmin (lambda={lambda_slack:g}) {plot_data['scenario_id']}")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "tank_temp_soft_15_vs_60.png"), dpi=150)
    plt.close(fig)


def run_soft_tank(time_index, p15, p60, tank_scenarios, tank_params, lambda_slack):
    targets = {"TANK_family4_4showers", "TANK_family4_6showers"}
    selected = [sc for sc in tank_scenarios if sc["scenario_id"] in targets]
    if not selected:
        selected = list(tank_scenarios)

    results = []
    plot_data = None
    fallback_plot = None
    for sc in selected:
        daily_kwh = sc.get("daily_hotwater_kwh")
        if daily_kwh is not None and not np.isnan(daily_kwh):
            validate_daily_draw(time_index, sc["draw_kwh"], daily_kwh, sc["scenario_id"])

        res_15 = optimize_tank_softmin(
            p15,
            time_index,
            sc["draw_kwh"],
            tank_params,
            dt_hours=DT_HOURS,
            hourly_control=False,
            lambda_slack=lambda_slack,
        )
        res_60 = optimize_tank_softmin(
            p60,
            time_index,
            sc["draw_kwh"],
            tank_params,
            dt_hours=DT_HOURS,
            hourly_control=True,
            lambda_slack=lambda_slack,
        )

        delta_soft_eur, delta_soft_pct = _compute_delta(
            res_60["cost"],
            res_15["cost"],
            scenario_id=f"{sc['scenario_id']}[SOFT]",
            method="OPT",
        )
        results.append(
            {
                "method": "OPT_SOFTMIN",
                "device": sc["device"],
                "scenario_id": sc["scenario_id"],
                "lambda_slack": lambda_slack,
                "cost_15min_soft": res_15["cost"],
                "cost_60min_soft": res_60["cost"],
                "delta_soft": delta_soft_eur,
                "delta_soft_pct": delta_soft_pct,
                "slack_sum_15": res_15.get("slack_sum", np.nan),
                "slack_max_15": res_15.get("slack_max", np.nan),
                "min_temp_15": res_15.get("min_temp", np.nan),
                "slack_sum_60": res_60.get("slack_sum", np.nan),
                "slack_max_60": res_60.get("slack_max", np.nan),
                "min_temp_60": res_60.get("min_temp", np.nan),
                "status_15min": res_15.get("status", ""),
                "status_60min": res_60.get("status", ""),
                "row_status": _row_status(res_15.get("status", ""), res_60.get("status", "")),
                "daily_hotwater_kwh": sc.get("daily_hotwater_kwh", np.nan),
                "events": sc.get("events", ""),
            }
        )

        if (
            _is_feasible(res_15.get("status", ""))
            and _is_feasible(res_60.get("status", ""))
            and res_15.get("Temp") is not None
            and res_60.get("Temp") is not None
        ):
            fallback_plot = {
                "time": pd.DatetimeIndex(time_index),
                "T_15": res_15["Temp"],
                "T_60": res_60["Temp"],
                "scenario_id": sc["scenario_id"],
            }
        if (
            sc["scenario_id"] == "TANK_family4_6showers"
            and _is_feasible(res_15.get("status", ""))
            and _is_feasible(res_60.get("status", ""))
            and res_15.get("Temp") is not None
            and res_60.get("Temp") is not None
        ):
            plot_data = {
                "time": pd.DatetimeIndex(time_index),
                "T_15": res_15["Temp"],
                "T_60": res_60["Temp"],
                "scenario_id": sc["scenario_id"],
            }

        def _fmt(x, digits=2):
            if x is None or not np.isfinite(x):
                return "nan"
            return f"{x:.{digits}f}"

        print(
            "SOFT TANK (lambda="
            f"{lambda_slack:g}): {sc['scenario_id']} "
            f"delta={_fmt(delta_soft_eur, 2)}EUR "
            f"slack_sum_15={_fmt(res_15.get('slack_sum'), 4)} "
            f"slack_max_15={_fmt(res_15.get('slack_max'), 3)} "
            f"minT_15={_fmt(res_15.get('min_temp'), 2)}"
        )

    if plot_data is None and fallback_plot is not None:
        plot_data = fallback_plot

    return results, plot_data


def _sanitize_filename(name):
    safe = []
    for ch in str(name):
        if ch.isalnum() or ch in ("-", "_", "."):
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe)


def _hourly_max_diff(power_kw, time_index):
    times = pd.DatetimeIndex(time_index)
    hours = times.floor("h")
    max_diff = 0.0
    start = 0
    n = len(power_kw)
    while start < n:
        hour = hours[start]
        end = start + 1
        while end < n and hours[end] == hour:
            end += 1
        block = power_kw[start:end]
        if block.size > 0:
            diff = float(np.max(block) - np.min(block))
            if diff > max_diff:
                max_diff = diff
        start = end
    return max_diff


def _price_hour_check(time_index, p15, p60):
    times = pd.DatetimeIndex(time_index)
    hours = times.floor("h")
    unique_hours = pd.Index(hours).unique()
    if unique_hours.empty:
        return None
    target = None
    preferred = pd.Timestamp("2025-10-07 00:00")
    if preferred in unique_hours:
        target = preferred
    else:
        target = unique_hours[0]
    mask = hours == target
    if np.sum(mask) != 4:
        return {"hour": target, "p60": np.nan, "p15_mean": np.nan, "diff": np.nan, "pass": False}
    p15_mean = float(np.mean(p15[mask]))
    p60_val = float(np.mean(p60[mask]))
    diff = abs(p60_val - p15_mean)
    return {
        "hour": target,
        "p60": p60_val,
        "p15_mean": p15_mean,
        "diff": diff,
        "pass": diff < 1e-9,
    }


def export_audit(
    ev_scenario,
    tank_scenario,
    time_index,
    p15,
    p60,
    tank_params,
    audit_n,
    out_dir,
):
    lines = []

    if ev_scenario is not None:
        start_idx = ev_scenario["start_idx"]
        end_idx = ev_scenario["end_idx"]
        times = pd.DatetimeIndex(time_index[start_idx:end_idx])
        prices_15 = p15[start_idx:end_idx]
        prices_60 = p60[start_idx:end_idx]

        res_15 = optimize_ev(
            prices_15,
            times,
            pmax=ev_scenario["pmax"],
            eta=ev_scenario["eta"],
            e_need=ev_scenario["e_need"],
            dt_hours=DT_HOURS,
            hourly_control=False,
        )
        res_60 = optimize_ev(
            prices_60,
            times,
            pmax=ev_scenario["pmax"],
            eta=ev_scenario["eta"],
            e_need=ev_scenario["e_need"],
            dt_hours=DT_HOURS,
            hourly_control=True,
        )
        P_15 = res_15.get("P")
        P_60 = res_60.get("P")

        df_ev = pd.DataFrame(
            {
                "Time": times,
                "price_15": prices_15,
                "price_60": prices_60,
                "P_15": P_15,
                "P_60": P_60,
                "E_15": P_15 * DT_HOURS if P_15 is not None else np.nan,
                "E_60": P_60 * DT_HOURS if P_60 is not None else np.nan,
            }
        )
        df_ev_out = df_ev.head(audit_n) if audit_n and audit_n > 0 else df_ev
        ev_name = _sanitize_filename(ev_scenario["scenario_id"])
        df_ev_out.to_csv(os.path.join(out_dir, f"audit_ev_{ev_name}.csv"), index=False)

        e_grid = ev_scenario["e_need"] / ev_scenario["eta"]
        e_del_15 = float(np.sum(P_15) * DT_HOURS) if P_15 is not None else np.nan
        e_del_60 = float(np.sum(P_60) * DT_HOURS) if P_60 is not None else np.nan
        ev_energy_pass = (
            np.isfinite(e_del_15)
            and np.isfinite(e_del_60)
            and abs(e_del_15 - e_grid) < 1e-3
            and abs(e_del_60 - e_grid) < 1e-3
        )
        lines.append(
            f"EV Energy delivered: E_grid={e_grid:.6f} "
            f"E15={e_del_15:.6f} E60={e_del_60:.6f} "
            f"PASS={ev_energy_pass}"
        )

        if P_60 is not None:
            max_diff = _hourly_max_diff(P_60, times)
            lines.append(f"EV Hourly control max diff: {max_diff:.8f} PASS={max_diff < 1e-6}")
        else:
            lines.append("EV Hourly control max diff: n/a PASS=False")

        if P_15 is not None:
            cost15_re = float(np.sum(prices_15 * P_15) * DT_HOURS)
            cost15_diff = abs(cost15_re - float(res_15.get("cost", np.nan)))
        else:
            cost15_re = np.nan
            cost15_diff = np.nan
        if P_60 is not None:
            cost60_re = float(np.sum(prices_60 * P_60) * DT_HOURS)
            cost60_diff = abs(cost60_re - float(res_60.get("cost", np.nan)))
        else:
            cost60_re = np.nan
            cost60_diff = np.nan
        lines.append(
            f"EV Cost recompute: cost15={cost15_re:.6f} diff={cost15_diff:.8f} "
            f"cost60={cost60_re:.6f} diff={cost60_diff:.8f} "
            f"PASS={(cost15_diff < 1e-6) and (cost60_diff < 1e-6)}"
        )
    else:
        lines.append("EV audit: not available")

    if tank_scenario is not None:
        times = pd.DatetimeIndex(time_index)
        res_15 = optimize_tank(
            p15,
            time_index,
            tank_scenario["draw_kwh"],
            tank_params,
            dt_hours=DT_HOURS,
            hourly_control=False,
        )
        res_60 = optimize_tank(
            p60,
            time_index,
            tank_scenario["draw_kwh"],
            tank_params,
            dt_hours=DT_HOURS,
            hourly_control=True,
        )
        P_15 = res_15.get("P")
        P_60 = res_60.get("P")
        T_15 = res_15.get("T")
        T_60 = res_60.get("T")
        temp_15 = T_15[:-1] if T_15 is not None else None
        temp_60 = T_60[:-1] if T_60 is not None else None

        df_tank = pd.DataFrame(
            {
                "Time": times,
                "price_15": p15,
                "price_60": p60,
                "draw_kwh": tank_scenario["draw_kwh"],
                "P_15": P_15,
                "P_60": P_60,
                "Temp_15": temp_15,
                "Temp_60": temp_60,
            }
        )
        df_tank_out = df_tank.head(audit_n) if audit_n and audit_n > 0 else df_tank
        tank_name = _sanitize_filename(tank_scenario["scenario_id"])
        df_tank_out.to_csv(os.path.join(out_dir, f"audit_tank_{tank_name}.csv"), index=False)

        if T_15 is not None:
            min_temp = float(np.min(T_15))
            max_temp = float(np.max(T_15))
            temp_pass = (min_temp >= tank_params["tmin"] - 1e-3) and (
                max_temp <= tank_params["tmax"] + 1e-3
            )
            lines.append(
                f"Tank Temp bounds (15): min={min_temp:.3f} max={max_temp:.3f} PASS={temp_pass}"
            )
        else:
            lines.append("Tank Temp bounds (15): n/a PASS=False")

        if P_60 is not None:
            max_diff = _hourly_max_diff(P_60, times)
            lines.append(f"Tank Hourly control max diff: {max_diff:.8f} PASS={max_diff < 1e-6}")
        else:
            lines.append("Tank Hourly control max diff: n/a PASS=False")

        draw_series = pd.Series(tank_scenario["draw_kwh"], index=times)
        top_draws = draw_series.sort_values(ascending=False).head(10)
        lines.append("Tank draw top-10 (time, kWh):")
        for ts, val in top_draws.items():
            lines.append(f"  {ts}: {val:.3f}")
    else:
        lines.append("Tank audit: not available")

    price_check = _price_hour_check(time_index, p15, p60)
    if price_check is not None:
        lines.append(
            f"Price hour check {price_check['hour']}: p60={price_check['p60']:.6f} "
            f"mean_p15={price_check['p15_mean']:.6f} diff={price_check['diff']:.10f} "
            f"PASS={price_check['pass']}"
        )

    with open(os.path.join(out_dir, "audit_checks.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def plot_prices(time_index, p15, p60, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(time_index, p15, label="15-min price")
    ax.plot(time_index, p60, label="Hourly avg")
    ax.set_ylabel("EUR/kWh")
    ax.set_title("Price series")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "prices_15_vs_60.png"), dpi=150)
    plt.close(fig)


def plot_ev_power(plot_data, out_dir):
    if plot_data is None:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.step(plot_data["time"], plot_data["P_15"], where="post", label="15-min control")
    ax.step(
        plot_data["time"],
        plot_data["P_60"],
        where="post",
        label="60-min control",
    )
    ax.set_ylabel("kW")
    ax.set_title(f"EV charging power ({plot_data['scenario_id']})")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "ev_power_15_vs_60.png"), dpi=150)
    plt.close(fig)


def plot_tank_temp(plot_data, out_dir):
    if plot_data is None:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = pd.DatetimeIndex(plot_data["time"])
    temp_times = times.append(
        pd.DatetimeIndex([times[-1] + pd.Timedelta(minutes=15)])
    )

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(temp_times, plot_data["T_15"], label="15-min control")
    ax.plot(temp_times, plot_data["T_60"], label="60-min control")
    ax.set_ylabel("C")
    ax.set_title(f"Tank temperature ({plot_data['scenario_id']})")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "tank_temp_15_vs_60.png"), dpi=150)
    plt.close(fig)


def print_top_deltas(results_df, top_n=5):
    if results_df.empty or "delta_eur" not in results_df.columns:
        return
    df = results_df.copy()
    if "method" in df.columns:
        df = df[df["method"] == "OPT"]
    if "row_status" in df.columns:
        df = df[df["row_status"] == "ok"]
    df = df.dropna(subset=["delta_eur"])
    if df.empty:
        return
    df["abs_delta"] = df["delta_eur"].abs()
    df = df.sort_values("abs_delta", ascending=False).head(top_n)
    print("Top delta_eur scenarios:")
    for _, row in df.iterrows():
        print(
            f"{row['device']} {row['scenario_id']} delta_eur={row['delta_eur']:.4f} "
            f"delta_pct={row['delta_pct']:.2f}"
        )


def print_ev_aggregate_stats(results_df):
    if results_df.empty:
        return
    ev = results_df[results_df["device"] == "EV"].copy()
    if "method" in ev.columns:
        ev = ev[ev["method"] == "OPT"]
    if "row_status" in ev.columns:
        ev = ev[ev["row_status"] == "ok"]
    ev = ev.dropna(subset=["delta_eur"])
    if ev.empty:
        return
    print("EV delta_eur stats by Pmax/E_need:")
    grouped = ev.groupby(["pmax", "e_need"])
    for (pmax, e_need), group in grouped:
        values = group["delta_eur"].to_numpy(dtype=float)
        if values.size == 0:
            continue
        mean = float(np.mean(values))
        median = float(np.median(values))
        vmin = float(np.min(values))
        vmax = float(np.max(values))
        print(
            f"Pmax={pmax:g} kW, E_need={e_need:g} kWh: "
            f"mean={mean:.4f}, median={median:.4f}, min={vmin:.4f}, max={vmax:.4f}, n={values.size}"
        )


def print_ev_monthly_totals(results_df):
    if results_df.empty:
        return
    ev = results_df[results_df["device"] == "EV"].copy()
    if "method" in ev.columns:
        ev = ev[ev["method"] == "OPT"]
    if "row_status" in ev.columns:
        ev = ev[ev["row_status"] == "ok"]
    ev = ev.dropna(subset=["cost_15min", "cost_60min", "delta_eur"])
    if ev.empty:
        return
    print("EV monthly totals (OPT):")
    grouped = ev.groupby(["pmax", "e_need"])
    for (pmax, e_need), group in grouped:
        sum_cost15 = float(group["cost_15min"].sum())
        sum_cost60 = float(group["cost_60min"].sum())
        sum_delta = float(group["delta_eur"].sum())
        pct = (sum_delta / sum_cost60 * 100.0) if abs(sum_cost60) > 1e-12 else np.nan
        print(
            f"Pmax={pmax:g}, E={e_need:g}: "
            f"cost15={sum_cost15:.2f}€, cost60={sum_cost60:.2f}€, "
            f"delta={sum_delta:.2f}€ ({pct:.1f}%)"
        )


def main():
    parser = argparse.ArgumentParser(description="EV and tank optimization comparison")
    parser.add_argument("--input", required=True, help="Path to price CSV")
    parser.add_argument("--out", default="outputs", help="Output directory")
    parser.add_argument("--expected-rows", type=int, default=None, help="Optional row count check")
    parser.add_argument("--run-soft-tank", action="store_true", help="Run soft Tmin tank report")
    parser.add_argument("--lambda-slack", type=float, default=100.0, help="Slack penalty weight")
    parser.add_argument("--audit", action="store_true", help="Run audit export")
    parser.add_argument("--audit-scenario", type=str, default="", help="Scenario id to audit")
    parser.add_argument("--audit-n", type=int, default=96, help="Rows in audit CSV output")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    df, p15 = read_prices(args.input, expected_rows=args.expected_rows)
    p60 = make_hourly_price_repeated(df)
    time_index = pd.DatetimeIndex(df["Time"])

    n_rows = len(df)
    start_ts = df["Time"].iloc[0]
    end_ts = df["Time"].iloc[-1]
    days = df["Time"].dt.normalize().nunique()
    hours = n_rows / 4.0
    print(
        f"Loaded {n_rows} rows (15min). Range: {start_ts} -> {end_ts}. "
        f"Days={days}, Hours={hours:.2f}."
    )
    if n_rows % 96 != 0:
        print("Warning: rows not multiple of 96 (possible DST shift or missing data).")

    tank_params = {
        "pmax": 3.0,
        "eta": 1.0,
        "tmin": 50.0,
        "tmax": 65.0,
        "t0": 60.0,
        "tamb": 20.0,
        "k_loss": 0.004,
        "c_kwh_per_c": 200.0 * 4.186 / 3600.0,
    }

    ev_scenarios = build_ev_scenarios(time_index)
    tank_scenarios = build_tank_scenarios(time_index)

    print_profile_summary(ev_scenarios, tank_scenarios)

    results_opt, plot_data = run_all(
        time_index, p15, p60, ev_scenarios, tank_scenarios, tank_params
    )
    results_base = run_baselines(time_index, p15, p60, ev_scenarios, tank_scenarios, tank_params)

    results_opt_df = pd.DataFrame(results_opt)
    results_base_df = pd.DataFrame(results_base)
    results_df = pd.concat([results_opt_df, results_base_df], ignore_index=True, sort=False)
    if not results_df.empty and {"method", "device", "scenario_id"}.issubset(results_df.columns):
        results_df = results_df.sort_values(["method", "device", "scenario_id"]).reset_index(drop=True)

    results_summary_path = os.path.join(args.out, "results_summary.csv")
    results_df.to_csv(results_summary_path, index=False)
    results_opt_df.to_csv(os.path.join(args.out, "results_opt.csv"), index=False)
    results_base_df.to_csv(os.path.join(args.out, "results_baseline.csv"), index=False)

    plot_prices(time_index, p15, p60, args.out)
    plot_ev_power(plot_data["ev"], args.out)
    plot_tank_temp(plot_data["tank"], args.out)

    print_top_deltas(results_opt_df, top_n=5)
    print_ev_aggregate_stats(results_opt_df)
    print_ev_monthly_totals(results_opt_df)

    if args.run_soft_tank:
        soft_rows, soft_plot = run_soft_tank(
            time_index, p15, p60, tank_scenarios, tank_params, args.lambda_slack
        )
        soft_df = pd.DataFrame(soft_rows)
        soft_path = os.path.join(args.out, "results_tank_soft.csv")
        soft_df.to_csv(soft_path, index=False)
        plot_tank_temp_soft(soft_plot, args.out, args.lambda_slack)

    if args.audit:
        ev_scenario = None
        tank_scenario = None
        if args.audit_scenario:
            for sc in ev_scenarios:
                if sc["scenario_id"] == args.audit_scenario:
                    ev_scenario = sc
                    break
            for sc in tank_scenarios:
                if sc["scenario_id"] == args.audit_scenario:
                    tank_scenario = sc
                    break
            if ev_scenario is None and tank_scenario is None:
                print(f"Audit scenario not found: {args.audit_scenario}")

        if ev_scenario is None:
            ev_candidates = results_opt_df[
                (results_opt_df["device"] == "EV") & (results_opt_df["row_status"] == "ok")
            ].dropna(subset=["delta_eur"])
            if not ev_candidates.empty:
                ev_id = ev_candidates.loc[ev_candidates["delta_eur"].abs().idxmax(), "scenario_id"]
                ev_scenario = next(
                    (sc for sc in ev_scenarios if sc["scenario_id"] == ev_id), None
                )
        if ev_scenario is None and ev_scenarios:
            ev_scenario = ev_scenarios[0]

        tank_scenario = tank_scenario
        if tank_scenario is None:
            preferred = next(
                (sc for sc in tank_scenarios if sc["scenario_id"] == "TANK_family4_6showers"),
                None,
            )
            if preferred is not None:
                tank_scenario = preferred
        if tank_scenario is None:
            tank_candidates = results_opt_df[
                (results_opt_df["device"] == "TANK") & (results_opt_df["row_status"] == "ok")
            ].dropna(subset=["delta_eur"])
            if not tank_candidates.empty:
                tank_id = tank_candidates.loc[
                    tank_candidates["delta_eur"].abs().idxmax(), "scenario_id"
                ]
                tank_scenario = next(
                    (sc for sc in tank_scenarios if sc["scenario_id"] == tank_id), None
                )
        if tank_scenario is None and tank_scenarios:
            tank_scenario = tank_scenarios[0]

        export_audit(
            ev_scenario,
            tank_scenario,
            time_index,
            p15,
            p60,
            tank_params,
            args.audit_n,
            args.out,
        )


if __name__ == "__main__":
    main()
