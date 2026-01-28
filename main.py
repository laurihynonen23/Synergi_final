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


def _compute_delta(cost_60, cost_15, scenario_id=""):
    if np.isnan(cost_60) or np.isnan(cost_15):
        return np.nan, np.nan
    delta = cost_60 - cost_15
    if np.isfinite(delta) and abs(delta) < EPS_EUR:
        return 0.0, 0.0
    if np.isfinite(delta) and delta < -1e-4:
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
            res_60["cost"], res_15["cost"], scenario_id=sc["scenario_id"]
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
            res_60["cost"], res_15["cost"], scenario_id=sc["scenario_id"]
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
                res_60["cost"], res_15["cost"], scenario_id=f"{sc['scenario_id']}[{method}]"
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
                res_60["cost"], res_15["cost"], scenario_id=f"{sc['scenario_id']}[{method}]"
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
            res_60["cost"], res_15["cost"], scenario_id=f"{sc['scenario_id']}[SOFT]"
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


def main():
    parser = argparse.ArgumentParser(description="EV and tank optimization comparison")
    parser.add_argument("--input", required=True, help="Path to price CSV")
    parser.add_argument("--out", default="outputs", help="Output directory")
    parser.add_argument("--run-soft-tank", action="store_true", help="Run soft Tmin tank report")
    parser.add_argument("--lambda-slack", type=float, default=100.0, help="Slack penalty weight")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    df, p15 = read_prices(args.input)
    p60 = make_hourly_price_repeated(df)
    time_index = pd.DatetimeIndex(df["Time"])

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

    if args.run_soft_tank:
        soft_rows, soft_plot = run_soft_tank(
            time_index, p15, p60, tank_scenarios, tank_params, args.lambda_slack
        )
        soft_df = pd.DataFrame(soft_rows)
        soft_path = os.path.join(args.out, "results_tank_soft.csv")
        soft_df.to_csv(soft_path, index=False)
        plot_tank_temp_soft(soft_plot, args.out, args.lambda_slack)


if __name__ == "__main__":
    main()
