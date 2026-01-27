import numpy as np
import pandas as pd


def _select_dates(candidate_dates, max_days):
    if max_days is None or len(candidate_dates) <= max_days:
        return candidate_dates
    idx = np.linspace(0, len(candidate_dates) - 1, max_days, dtype=int)
    return [candidate_dates[i] for i in idx]


def build_ev_scenarios(
    time_index,
    pmax_list=(3.6, 11.0),
    e_need_list=(10.0, 20.0, 30.0),
    eta=0.9,
    plug_in_hour=17,
    depart_hour=8,
    max_days=None,
):
    times = pd.DatetimeIndex(time_index)
    time_to_idx = {ts: i for i, ts in enumerate(times)}
    dates = sorted(pd.Index(times.normalize()).unique())

    candidates = []
    for day in dates:
        plug_in = day + pd.Timedelta(hours=plug_in_hour)
        depart = day + pd.Timedelta(days=1, hours=depart_hour)
        if plug_in in time_to_idx and depart in time_to_idx:
            candidates.append(day)

    candidates = _select_dates(candidates, max_days)
    scenarios = []
    home_window = "17:00-08:00"
    for day in candidates:
        plug_in = day + pd.Timedelta(hours=plug_in_hour)
        depart = day + pd.Timedelta(days=1, hours=depart_hour)
        start_idx = time_to_idx[plug_in]
        end_idx = time_to_idx[depart]
        if end_idx <= start_idx:
            continue
        for pmax in pmax_list:
            for e_need in e_need_list:
                scenario_id = f"EV_{plug_in.date()}_P{pmax:g}_E{e_need:g}"
                scenarios.append(
                    {
                        "device": "EV",
                        "scenario_id": scenario_id,
                        "start_idx": start_idx,
                        "end_idx": end_idx,
                        "pmax": float(pmax),
                        "eta": float(eta),
                        "e_need": float(e_need),
                        "start_time": plug_in,
                        "end_time": depart,
                        "home_window": home_window,
                    }
                )
    return scenarios


def add_spread_event(draw_series, time_index, base_time, total_kwh, slots=2):
    per = total_kwh / slots
    for i in range(slots):
        t = base_time + pd.Timedelta(minutes=15 * i)
        if t in time_index:
            draw_series.loc[t] += per


def build_draw_profile(time_index, daily_events):
    times = pd.DatetimeIndex(time_index)
    dates = sorted(pd.Index(times.normalize()).unique())
    draw = pd.Series(0.0, index=times)
    for day in dates:
        for hour, minute, total_kwh, slots in daily_events:
            base = day + pd.Timedelta(hours=hour, minutes=minute)
            add_spread_event(draw, times, base, total_kwh, slots=slots)
    return draw.to_numpy(dtype=float)


def build_family4_tank_scenarios(time_index):
    shower_kwh = 1.5
    scenarios = []

    events_4 = [
        (6, 30, shower_kwh, 2),
        (7, 0, shower_kwh, 2),
        (19, 0, shower_kwh, 2),
        (20, 0, shower_kwh, 2),
    ]
    events_6 = [
        (6, 30, shower_kwh, 2),
        (7, 0, shower_kwh, 2),
        (19, 0, shower_kwh, 2),
        (19, 30, shower_kwh, 2),
        (20, 0, shower_kwh, 2),
        (20, 30, shower_kwh, 2),
    ]

    for name, events, daily_kwh in (
        ("family4_4showers", events_4, 6.0),
        ("family4_6showers", events_6, 9.0),
    ):
        draw_kwh = build_draw_profile(time_index, events)
        scenarios.append(
            {
                "device": "TANK",
                "scenario_id": f"TANK_{name}",
                "draw_kwh": draw_kwh,
                "profile": name,
                "daily_hotwater_kwh": daily_kwh,
                "events": events,
            }
        )

    return scenarios


def build_tank_scenarios(time_index):
    profiles = [
        {
            "name": "low",
            "events": [(7, 0, 1.5, 2), (20, 0, 1.5, 2)],
            "daily_hotwater_kwh": 3.0,
        },
        {
            "name": "medium",
            "events": [(7, 0, 3.0, 2), (20, 0, 3.0, 2)],
            "daily_hotwater_kwh": 6.0,
        },
    ]

    scenarios = []
    for profile in profiles:
        draw = build_draw_profile(time_index, profile["events"])
        scenarios.append(
            {
                "device": "TANK",
                "scenario_id": f"TANK_{profile['name']}",
                "draw_kwh": draw,
                "profile": profile["name"],
                "daily_hotwater_kwh": profile["daily_hotwater_kwh"],
                "events": profile["events"],
            }
        )

    scenarios.extend(build_family4_tank_scenarios(time_index))
    return scenarios
