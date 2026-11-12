"""Scenario-conditional reconstruction with availability and group selection.

This module reads no passenger reference files. It estimates group arrival rates
from positive booking operations, and calibrates a WTP intercept against an
EXTERNAL primary-demand seat total. Coefficients/sigma are fixed assumptions or
independently supplied calibration results. In particular, there is no claim
that these quantities are identified from two booking histories.

There is one offered product per cabin and time. Real interval data must cover
the entire supplied horizon. With no offers, purchase-derived price paths and
unlimited availability are explicit assumptions. No substitution between cabins
or products is inferred. Cancellation prices never update an offered price.
"""

from dataclasses import dataclass
from math import erfc, exp, log, sqrt
from statistics import NormalDist
from typing import Mapping

import numpy as np
import pandas as pd


PERIOD_ORDER = ["91+", "31–90", "15–30", "8–14", "0–7"]
STRATUM_KEYS = ["flight", "cabin", "period"]
PATH_METHODS = ("last", "midpoint", "next")
REASONS = ("availability", "budget", "other")
_NORMAL = NormalDist()


@dataclass
class ModelConfig:
    """Fixed behavioral assumptions, not fitted on evaluation passenger files.

    log(W) = mu_f + beta_time*log((1+dtd)/31) + cabin_offsets[c]
             + beta_qty*log(q) + sigma*Z.

    qty_prior is an optional ALL-DEMAND group-size distribution. Accepted forms:
      {1: .7, 2: .3}; {'C': {...}, 'Y': {...}};
      {(flight, cabin): {...}}; {(flight, cabin, period): {...}}.
    An external prior is used directly, without buyer-mixture contamination.
    Without one, buyer quantities are extrapolated with optional pseudo-count
    smoothing toward cabin/global buyers; that selection assumption is flagged.
    """
    sigma: float = 0.45
    beta_time: float = 0.0
    beta_qty: float = 0.0
    cabin_offsets: object = None
    other_reject_probability: float = 0.0
    qty_values: tuple = (1, 2, 3, 4, 5, 6)
    qty_prior: object = None
    qty_smoothing: float = 0.0


@dataclass
class ModelInputsV2:
    bookings: pd.DataFrame
    purchases: pd.DataFrame
    price_segments: pd.DataFrame
    strata: pd.DataFrame
    assumptions: dict


def _utc(value):
    t = pd.Timestamp(value)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _period(dtd):
    if dtd < 0:
        raise ValueError("An interval falls after the departure calendar day.")
    return "0–7" if dtd <= 7 else "8–14" if dtd <= 14 else "15–30" if dtd <= 30 else "31–90" if dtd <= 90 else "91+"


def _bool_values(values):
    def convert(value):
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, float, np.integer, np.floating)) and value in (0, 1):
            return bool(value)
        if isinstance(value, str) and value.lower().strip() in ("true", "false", "0", "1"):
            return value.lower().strip() in ("true", "1")
        raise ValueError(f"Invalid open flag: {value!r}")
    return values.map(convert)


def _split_intervals(intervals, departure):
    result = []
    for row in intervals.to_dict("records"):
        cursor = row["start"]
        while cursor < row["end"]:
            stop = min(row["end"], cursor.normalize() + pd.Timedelta(days=1))
            dtd = int((departure - cursor.normalize()).days)
            result.append({**row, "start": cursor, "end": stop, "dtd": dtd,
                           "period": _period(dtd), "duration_days": (stop-cursor).total_seconds()/86400})
            cursor = stop
    output = pd.DataFrame(result).reset_index(drop=True)
    output["segment_id"] = np.arange(len(output), dtype=int)
    return output


def prepare_inputs_v2(bookings, departure_date, horizon_start, horizon_end, offers=None, path_method="last",
                      *, allow_empty_purchases=False):
    """Validate booking operations and construct/validate complete offer intervals.

    offers columns: flight,cabin,start,end,price,open,available_seats; optional rbd.
    Intervals are [start,end), nonoverlapping, gap-free and must exactly cover
    [horizon_start,horizon_end) for each supplied flight/cabin. available_seats is
    the supplied offer-state at that interval, not an invented physical capacity.
    Each observed purchase must match an open interval, price and sufficient
    group availability. When an rbd is supplied it is checked too.

    horizon_end may be next UTC midnight to include all of DtD=0. Real departure
    hours are not inferred. Price-change times inferred from purchases are never
    called real offer history. Cancel-only cabins need actual offer intervals.
    """
    if path_method not in PATH_METHODS:
        raise ValueError(f"path_method must be one of {PATH_METHODS}")
    start, end, departure = _utc(horizon_start), _utc(horizon_end), _utc(departure_date).normalize()
    if not start < end or end > departure + pd.Timedelta(days=1):
        raise ValueError("Invalid horizon: include only dates through the departure calendar day.")
    required = {"flight", "timestamp", "cabin", "commercial", "price"}
    if required-set(bookings.columns):
        raise ValueError(f"Missing booking columns: {sorted(required-set(bookings.columns))}")
    frame = bookings.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    frame["commercial"] = pd.to_numeric(frame["commercial"], errors="raise")
    if (frame[["flight", "cabin", "timestamp", "commercial"]].isna().any().any()
            or not np.isfinite(frame.commercial).all()
            or frame.commercial.eq(0).any() or frame.commercial.mod(1).ne(0).any()):
        raise ValueError("Booking keys must be present and commercial a finite nonzero integer.")
    frame["commercial"] = frame.commercial.astype(int)
    if frame.timestamp.lt(start).any() or frame.timestamp.ge(end).any():
        raise ValueError("The specified horizon excludes booking operations.")
    frame["qty"] = frame.commercial.abs()
    frame["dtd"] = (departure-frame.timestamp.dt.normalize()).dt.days
    frame["period"] = frame.dtd.map(_period)
    if "source_row" not in frame:
        frame["source_row"] = frame.groupby("flight", sort=False).cumcount()+2
    purchases = frame.loc[frame.commercial.gt(0)].copy()
    if purchases.empty and not (allow_empty_purchases and offers is not None):
        raise ValueError("No purchases; a demand scale cannot be inferred from bookings alone.")
    purchases["price"] = pd.to_numeric(purchases["price"], errors="raise")
    if not np.isfinite(purchases.price).all() or purchases.price.le(0).any():
        raise ValueError("Purchase prices must be positive and finite.")

    if offers is None:
        rows = []
        for (flight, cabin), group in purchases.groupby(["flight", "cabin"], sort=True):
            if group.groupby("timestamp").price.nunique().gt(1).any():
                raise ValueError("Simultaneous different prices cannot define a single-offer path.")
            states = group.sort_values("timestamp").drop_duplicates("timestamp")
            times = states.timestamp.tolist()
            transitions = [after if path_method == "last" else before if path_method == "next"
                           else before+(after-before)/2 for before, after in zip(times, times[1:])]
            boundaries = [start]+transitions+[end]
            for left, right, state in zip(boundaries, boundaries[1:], states.to_dict("records")):
                if right <= left:
                    continue
                rows.append({"flight": flight, "cabin": cabin, "start": left, "end": right,
                             "price": float(state["price"]), "open": True,
                             "available_seats": np.inf, "rbd": state.get("rbd", pd.NA),
                             "offer_source": f"assumed_{path_method}_purchase_prices"})
        intervals = pd.DataFrame(rows)
        source_name = "purchase_derived_assumption"
    else:
        needed = {"flight", "cabin", "start", "end", "price", "open", "available_seats"}
        if needed-set(offers.columns):
            raise ValueError(f"Missing offers columns: {sorted(needed-set(offers.columns))}")
        intervals = offers.copy()
        intervals["start"] = pd.to_datetime(intervals.start, utc=True, errors="raise")
        intervals["end"] = pd.to_datetime(intervals.end, utc=True, errors="raise")
        intervals["open"] = _bool_values(intervals.open)
        intervals["price"] = pd.to_numeric(intervals.price, errors="coerce")
        intervals["available_seats"] = pd.to_numeric(intervals.available_seats, errors="raise")
        if (not np.isfinite(intervals.available_seats).all() or intervals.available_seats.lt(0).any()
                or intervals.available_seats.mod(1).ne(0).any()):
            raise ValueError("Real available_seats must be finite nonnegative integers.")
        opened = intervals.open
        if (not np.isfinite(intervals.loc[opened, "price"]).all()
                or intervals.loc[opened, "price"].le(0).any()):
            raise ValueError("Every open offer needs a positive finite price.")
        if "rbd" not in intervals:
            intervals["rbd"] = pd.NA
        intervals["offer_source"] = "supplied_offer_intervals"
        source_name = "supplied_offer_intervals"

    if intervals.empty or intervals[["flight", "cabin", "start", "end"]].isna().any().any():
        raise ValueError("Offer intervals are empty or contain missing keys/times.")
    if intervals.end.le(intervals.start).any():
        raise ValueError("Offer intervals must have positive duration.")
    if set(intervals.flight)-set(frame.flight) and not allow_empty_purchases:
        raise ValueError("Offers contain a flight with no booking observations.")
    for (flight, cabin), group in intervals.groupby(["flight", "cabin"], sort=False):
        ordered = group.sort_values("start")
        if ordered.start.iloc[0] != start or ordered.end.iloc[-1] != end:
            raise ValueError(f"Offer history must exactly cover the complete horizon: {flight}/{cabin}.")
        if not (ordered.end.iloc[:-1].reset_index(drop=True) == ordered.start.iloc[1:].reset_index(drop=True)).all():
            raise ValueError(f"Offer history has a gap or overlap: {flight}/{cabin}.")
    pairs = set(map(tuple, intervals[["flight", "cabin"]].drop_duplicates().to_numpy()))
    missing_pairs = set(map(tuple, frame[["flight", "cabin"]].drop_duplicates().to_numpy()))-pairs
    if missing_pairs:
        raise ValueError(f"No offer history for flight/cabin: {sorted(missing_pairs)}")

    # Validate supplied offers only. Inferred paths deliberately do not assert
    # observed state at measure-zero switch points (especially next/midpoint).
    if offers is not None:
        for (flight, cabin), group in purchases.groupby(["flight", "cabin"], sort=False):
            history = intervals.loc[intervals.flight.eq(flight)&intervals.cabin.eq(cabin)].sort_values("start")
            # pandas can store otherwise identical UTC arrays in us or ns units;
            # integer search requires one common resolution.
            begins = history.start.dt.as_unit("ns").astype("int64").to_numpy()
            instants = group.timestamp.dt.as_unit("ns").astype("int64").to_numpy()
            indices = np.searchsorted(begins, instants, side="right")-1
            states = history.iloc[indices].reset_index(drop=True)
            events = group.reset_index(drop=True)
            valid = (states.open.to_numpy() & (events.qty.to_numpy() <= states.available_seats.to_numpy())
                     & np.isclose(events.price.to_numpy(), states.price.to_numpy(), rtol=0, atol=1e-7)
                     & (events.timestamp.to_numpy() < states.end.to_numpy()))
            if "rbd" in events:
                known = states.rbd.notna().to_numpy()
                valid &= ~known | states.rbd.astype(str).eq(events.rbd.astype(str)).to_numpy()
            if not valid.all():
                bad = events.loc[~valid, ["timestamp", "qty", "price"]].head(3).to_dict("records")
                raise ValueError(f"Purchases contradict supplied offers in {flight}/{cabin}: {bad}")

    segments = _split_intervals(intervals, departure)
    strata = segments.groupby(STRATUM_KEYS, as_index=False, sort=False).agg(exposure_days=("duration_days", "sum"))
    counts = purchases.groupby(STRATUM_KEYS, as_index=False).agg(
        purchase_events=("qty", "size"), purchased_seats=("qty", "sum"))
    strata = strata.merge(counts, on=STRATUM_KEYS, how="left", validate="one_to_one")
    strata[["purchase_events", "purchased_seats"]] = strata[["purchase_events", "purchased_seats"]].fillna(0).astype(int)
    rank = {period: i for i, period in enumerate(PERIOD_ORDER)}
    strata["_sort"] = strata.period.map(rank)
    strata = strata.sort_values(["flight", "cabin", "_sort"]).drop(columns="_sort").reset_index(drop=True)
    strata["stratum_id"] = np.arange(len(strata), dtype=int)
    return ModelInputsV2(frame, purchases, segments, strata, {
        "offer_source": source_name, "path_method": None if offers is not None else path_method,
        "horizon_start": start, "horizon_end": end, "departure_calendar_date": departure,
        "availability": "supplied_interval_state" if offers is not None else "assumed_unlimited",
        "single_product": "one_offer_per_cabin_no_cabin_substitution",
        "time_units": "UTC_calendar_DtD_including_day_zero",
        "amount_semantics": "assumed_per_seat_WTP_needs_confirmation",
        "rate_zero": "zero_sales_boundary_not_proof_of_zero_primary_demand",
    })


def _external_qty(prior, flight, cabin, period):
    if prior is None:
        return None
    if not isinstance(prior, Mapping):
        raise ValueError("qty_prior must be a dictionary or None.")
    if prior and all(np.isscalar(value) for value in prior.values()):
        return prior
    for key in ((flight, cabin, period), (flight, cabin), cabin, "global"):
        if key in prior:
            return prior[key]
    return None


def build_group_distributions_v2(inputs, config):
    """Potential-group-size probabilities with their provenance kept explicit."""
    q = np.asarray(config.qty_values, dtype=int)
    if len(q) == 0 or len(set(q)) != len(q) or (q <= 0).any() or any(int(v) != v for v in config.qty_values):
        raise ValueError("qty_values must be distinct positive integers.")
    if set(inputs.purchases.qty)-set(q):
        raise ValueError("Observed group size exceeds qty_values; explicitly extend the support.")
    if config.qty_smoothing < 0 or not np.isfinite(config.qty_smoothing):
        raise ValueError("qty_smoothing must be finite and nonnegative.")
    global_counts = inputs.purchases.qty.value_counts().reindex(q, fill_value=0).to_numpy(dtype=float)
    global_distribution = global_counts/global_counts.sum() if global_counts.sum() else np.zeros(len(q))
    rows = []
    for stratum in inputs.strata.itertuples(index=False):
        prior = _external_qty(config.qty_prior, stratum.flight, stratum.cabin, stratum.period)
        if prior is not None:
            if not isinstance(prior, Mapping) or set(prior)-set(q):
                raise ValueError("External qty prior must map supported quantities to probabilities.")
            weights = np.array([float(prior.get(int(value), 0)) for value in q])
            source = "external_all_demand_prior"
        else:
            if not global_counts.sum():
                raise ValueError("No purchases: supply an external all-demand qty_prior for every flight/cabin.")
            same = inputs.purchases.loc[inputs.purchases.flight.eq(stratum.flight)
                                        & inputs.purchases.cabin.eq(stratum.cabin)]
            local = same.loc[same.period.eq(stratum.period)]
            if len(local):
                counts = local.qty.value_counts().reindex(q, fill_value=0).to_numpy(dtype=float)
                base_counts = same.qty.value_counts().reindex(q, fill_value=0).to_numpy(dtype=float)
                base = base_counts/base_counts.sum()
                source = "buyer_stratum_extrapolation"
            elif len(same):
                counts = same.qty.value_counts().reindex(q, fill_value=0).to_numpy(dtype=float)
                base = global_distribution
                source = "buyer_cabin_fallback_extrapolation"
            else:
                counts = global_counts.copy()
                base = global_distribution
                source = "buyer_global_fallback_extrapolation"
            weights = counts+config.qty_smoothing*base
        if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError("Group-size probabilities must be finite, nonnegative and have positive sum.")
        probabilities = weights/weights.sum()
        for quantity, probability in zip(q, probabilities):
            rows.append({"flight": stratum.flight, "cabin": stratum.cabin, "period": stratum.period,
                         "stratum_id": stratum.stratum_id, "qty": int(quantity),
                         "probability": float(probability), "distribution_source": source,
                         "selection_assumption": prior is None})
    return pd.DataFrame(rows)


def _model_arrays(inputs, config):
    if not np.isfinite(config.sigma) or config.sigma <= 0:
        raise ValueError("sigma must be positive and finite.")
    if not np.isfinite(config.beta_time) or not np.isfinite(config.beta_qty):
        raise ValueError("Time and quantity coefficients must be finite.")
    r = float(config.other_reject_probability)
    if not np.isfinite(r) or r < 0 or r >= 1:
        raise ValueError("other_reject_probability must be in [0,1); r=1 cannot explain purchases.")
    offsets = {} if config.cabin_offsets is None else dict(config.cabin_offsets)
    if not all(np.isfinite(value) for value in offsets.values()):
        raise ValueError("Cabin offsets must be finite log-WTP offsets.")
    distributions = build_group_distributions_v2(inputs, config)
    segments = inputs.price_segments.merge(inputs.strata[STRATUM_KEYS+["stratum_id"]], on=STRATUM_KEYS, validate="many_to_one")
    marks = segments.merge(distributions[["stratum_id", "qty", "probability", "distribution_source"]],
                            on="stratum_id", how="left", validate="many_to_many")
    observed_q = inputs.purchases.merge(
        distributions[STRATUM_KEYS+["qty", "probability"]], on=STRATUM_KEYS+["qty"],
        how="left", validate="many_to_one")
    if observed_q.probability.isna().any() or observed_q.probability.le(0).any():
        raise ValueError("The supplied quantity distribution assigns zero probability to an observed purchase group.")
    index = marks.stratum_id.to_numpy(dtype=int)
    quantities = marks.qty.to_numpy(dtype=int)
    duration = marks.duration_days.to_numpy(dtype=float)
    probability = marks.probability.to_numpy(dtype=float)
    base = duration*probability
    available = marks.open.to_numpy(dtype=bool)&(quantities <= marks.available_seats.to_numpy(dtype=float))
    safe_price = np.where(marks.open.to_numpy(), marks.price.to_numpy(dtype=float), 1.0)
    log_offset = (config.beta_time*np.log((1+marks.dtd.to_numpy(dtype=float))/31)
                  + marks.cabin.map(offsets).fillna(0).to_numpy(dtype=float)
                  + config.beta_qty*np.log(quantities))
    threshold = np.log(safe_price)-log_offset
    return {"marks": marks, "index": index, "quantities": quantities,
            "base": base, "available": available.astype(float), "threshold": threshold,
            "config": config, "group_distributions": distributions, "r": r}


def _probabilities(arrays, mu):
    z = (arrays["threshold"]-mu)/(arrays["config"].sigma*sqrt(2))
    survival = np.fromiter((.5*erfc(float(value)) for value in z), dtype=float, count=len(z))
    a = arrays["available"]
    capacity = 1-a
    budget = a*(1-survival)
    other = a*survival*arrays["r"]
    buy = a*survival*(1-arrays["r"])
    return survival, capacity, budget, other, buy


def _compute_single(arrays, rows, mu):
    survival, capacity, budget, other, buy = _probabilities(arrays, mu)
    z = np.bincount(arrays["index"], weights=arrays["base"]*buy, minlength=len(rows))
    n = rows.purchase_events.to_numpy(dtype=float)
    impossible = (z <= 0)&(n > 0)
    if impossible.any():
        # Underflow far outside the calibrated bracket is numerically floored;
        # structural absence of an available Q must remain an explicit failure.
        structural = np.bincount(arrays["index"], weights=arrays["base"]*arrays["available"], minlength=len(rows))
        if ((structural <= 0)&(n > 0)).any():
            raise ValueError("A stratum with purchases has no available supported group size.")
    rates = np.divide(n, np.maximum(z, 1e-280), out=np.zeros_like(n), where=n > 0)
    potential = rates[arrays["index"]]*arrays["base"]
    hidden = potential*(capacity+budget+other)
    seats = hidden*arrays["quantities"]
    return {"survival": survival, "capacity": capacity, "budget": budget,
            "other": other, "buy": buy, "effective": z, "rates": rates,
            "potential": potential, "hidden": hidden, "hidden_seats": seats}


def evaluate_model_v2(inputs, intercept_by_flight, config=None):
    """Evaluate fixed mu_f, profiling group rates n_buy/integrated p_buy.

    Expected hidden GROUP quantities are selection-weighted by q and availability.
    A no-sales stratum keeps a boundary rate 0 but is marked unidentified. No
    positive arrival prior is silently added to closed or unobserved strata.
    """
    config = ModelConfig() if config is None else config
    full = _model_arrays(inputs, config)
    flights = inputs.strata.flight.unique().tolist()
    mu = dict(intercept_by_flight) if isinstance(intercept_by_flight, Mapping) else {f: float(intercept_by_flight) for f in flights}
    strata_parts, base_parts, reason_parts, parameter_rows = [], [], [], []
    for flight in flights:
        if flight not in mu or not np.isfinite(mu[flight]):
            raise ValueError("Provide a finite log-WTP intercept for every flight.")
        rows = inputs.strata.loc[inputs.strata.flight.eq(flight)].copy().reset_index(drop=True)
        old_ids = rows.stratum_id.to_numpy()
        remap = {old: i for i, old in enumerate(old_ids)}
        take = full["marks"].flight.eq(flight).to_numpy()
        arrays = {k: (v[take] if isinstance(v, np.ndarray) else v) for k, v in full.items() if k != "marks"}
        arrays["index"] = np.array([remap[i] for i in full["index"][take]], dtype=int)
        values = _compute_single(arrays, rows, float(mu[flight]))
        rows["effective_exposure_days"] = values["effective"]
        rows["lambda_groups_per_day"] = values["rates"]
        rows["identification"] = np.where(rows.purchase_events.gt(0), "model_conditional", "zero_sales_unidentified")
        raw_groups = np.bincount(arrays["index"], weights=values["hidden"], minlength=len(rows))
        raw_seats = np.bincount(arrays["index"], weights=values["hidden_seats"], minlength=len(rows))
        rows["expected_hidden_groups"] = raw_groups
        rows["raw_expected_hidden_seats"] = raw_seats
        rows["expected_primary_seats"] = rows.purchased_seats+raw_seats
        rows["hidden_mean_group_size"] = np.divide(raw_seats, raw_groups, out=np.full(len(rows), np.nan), where=raw_groups > 0)
        rows["hidden_seat_share"] = raw_seats/raw_seats.sum() if raw_seats.sum() else 0.0
        base = full["marks"].loc[take].copy()
        base["mu_log"] = float(mu[flight])
        base["wtp_log_location"] = (float(mu[flight])+config.beta_time*np.log((1+base.dtd)/31)
                                     + base.cabin.map(config.cabin_offsets or {}).fillna(0)
                                     + config.beta_qty*np.log(base.qty))
        for name in ("survival", "capacity", "budget", "other", "buy"):
            base[f"p_{name}"] = values[name]
        base["expected_primary_groups"] = values["potential"]
        base["expected_hidden_groups"] = values["hidden"]
        base["raw_expected_hidden_seats"] = values["hidden_seats"]
        base_parts.append(base)
        for reason in REASONS:
            marked = base.copy()
            marked["reason"] = reason
            probability_key = "capacity" if reason == "availability" else reason
            marked["reason_probability"] = values[probability_key]
            marked["expected_hidden_groups"] = values["potential"]*values[probability_key]
            marked["raw_expected_hidden_seats"] = marked.expected_hidden_groups*marked.qty
            # These long-format rows contain disjoint reason intensities. Do not
            # sum their duplicated primary-group intensities across reasons.
            marked = marked.drop(columns="expected_primary_groups")
            reason_parts.append(marked)
            rows[f"hidden_seats_{reason}"] = np.bincount(
                arrays["index"], weights=values["potential"]*values[probability_key]*arrays["quantities"], minlength=len(rows))
        strata_parts.append(rows)
        safe_median = exp(float(mu[flight])) if float(mu[flight]) < 709 else np.inf
        parameters = {"flight": flight, "mu_log": float(mu[flight]), "median_amount": safe_median,
                      "sigma": config.sigma, "beta_time": config.beta_time, "beta_qty": config.beta_qty,
                      "other_reject_probability": config.other_reject_probability,
                      "expected_primary_seats": float(rows.expected_primary_seats.sum()),
                      "raw_expected_hidden_seats": float(raw_seats.sum()),
                      "zero_sales_strata": int(rows.purchase_events.eq(0).sum()),
                      "min_buy_probability": float(values["buy"].min()),
                      "max_hidden_stratum_share": float(rows.hidden_seat_share.max())}
        parameter_rows.append(parameters)
    assumptions = dict(inputs.assumptions)
    assumptions.update({"wtp_family": "assumed_lognormal", "fixed_behavior_coefficients": True,
                        "qty_selection": "modelled_through_q_specific_purchase_probabilities",
                        "qty_prior": "external_or_explicit_buyer_extrapolation_see_group_distributions",
                        "reject_order": "availability_then_budget_then_other",
                        "within_segment_time": "uniform_given_piecewise_constant_daily_model",
                        "estimation": "profiled_unconditioned_group_rates_with_external_volume_anchor_not_exact_conditional_MLE"})
    return {"strata": pd.concat(strata_parts, ignore_index=True),
            "price_segments": inputs.price_segments.copy(),
            "parameters": pd.DataFrame(parameter_rows), "assumptions": assumptions,
            "marks": pd.concat(reason_parts, ignore_index=True),
            "base_marks": pd.concat(base_parts, ignore_index=True),
            "group_distributions": full["group_distributions"], "config": config}


def _allocate_integer(total, weights):
    weights = np.asarray(weights, dtype=float)
    if int(total) != total or total < 0 or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Invalid seat total or allocation weights.")
    if total == 0:
        return np.zeros(len(weights), dtype=int)
    if weights.sum() <= 0:
        raise ValueError("No positive weights for a positive hidden-seat total.")
    exact = int(total)*weights/weights.sum()
    answer = np.floor(exact).astype(int)
    left = int(total)-int(answer.sum())
    answer[np.argsort(-(exact-answer), kind="stable")[:left]] += 1
    return answer


def fit_model_v2(inputs, primary_demand_seats=300, config=None):
    """Calibrate only intercept_f to gross observed seats + expected hidden seats.

    For each flight, sigma/betas/availability/Q-prior/other-rejection remain fixed.
    Sixty bisections solve the external primary-volume anchor. Infeasible priors
    fail clearly; no arbitrary rescaling conceals incompatible behavior settings.
    Exact per-realization totals require sample_passengers_v2's separate CP
    conditioning. Integer stratum totals here are deterministic summary rounding.
    """
    config = ModelConfig() if config is None else config
    full = _model_arrays(inputs, config)
    flights = inputs.strata.flight.unique().tolist()
    targets = dict(primary_demand_seats) if isinstance(primary_demand_seats, Mapping) else {f: primary_demand_seats for f in flights}
    intercepts, statuses = {}, {}
    for flight in flights:
        rows = inputs.strata.loc[inputs.strata.flight.eq(flight)].copy().reset_index(drop=True)
        target = float(targets[flight])
        gross = int(rows.purchased_seats.sum())
        if not np.isfinite(target) or target != int(target) or target < gross:
            raise ValueError(f"Primary seats for {flight} must be an integer >= gross purchased seats {gross}.")
        take = full["marks"].flight.eq(flight).to_numpy()
        remap = {old: new for new, old in enumerate(rows.stratum_id)}
        arrays = {k: (v[take] if isinstance(v, np.ndarray) else v) for k, v in full.items() if k != "marks"}
        arrays["index"] = np.array([remap[i] for i in full["index"][take]], dtype=int)

        def total(mu):
            state = _compute_single(arrays, rows, mu)
            return gross+float(state["hidden_seats"].sum())

        # Analytic WTP->infinity feasibility, independent of finite brackets.
        infinite_state = _compute_single(arrays, rows, np.inf)
        lower_bound = gross+float(infinite_state["hidden_seats"].sum())
        if lower_bound > target+1e-6:
            raise ValueError(f"Infeasible fixed assumptions for {flight}: even unlimited WTP implies {lower_bound:.3f} primary seats > {target:.0f}.")
        margin = max(12.0, 10.0*float(config.sigma))
        lo, hi = float(arrays["threshold"].min()-margin), float(arrays["threshold"].max()+margin)
        for _ in range(12):
            if total(hi) <= target+1e-7:
                break
            hi += margin
        if total(lo) < target-1e-6:
            raise ValueError(f"Could not bracket the primary-volume anchor for {flight}.")
        if target == gross and abs(lower_bound-target) < 1e-7:
            intercepts[flight] = hi
            statuses[flight] = "no_hidden_volume_scale_lower_bound_only"
        else:
            for _ in range(60):
                middle = (lo+hi)/2
                if total(middle) > target:
                    lo = middle
                else:
                    hi = middle
            intercepts[flight] = (lo+hi)/2
            statuses[flight] = "calibrated_external_seat_total_fixed_behavior"
    result = evaluate_model_v2(inputs, intercepts, config)
    strata = result["strata"]
    strata["estimated_waste_seats"] = 0
    strata["normalization_factor"] = 1.0
    for flight in flights:
        take = strata.flight.eq(flight)
        target = int(targets[flight])-int(strata.loc[take, "purchased_seats"].sum())
        raw = strata.loc[take, "raw_expected_hidden_seats"].to_numpy()
        if abs(raw.sum()-target) > 1e-6:
            raise AssertionError("Calibrated expected hidden seats do not satisfy the known volume anchor.")
        strata.loc[take, "estimated_waste_seats"] = _allocate_integer(target, raw)
        strata.loc[take, "normalization_factor"] = target/raw.sum() if raw.sum() else 1.0
    strata["estimated_waste_seats"] = strata.estimated_waste_seats.astype(int)
    result["parameters"]["primary_demand_seats"] = result["parameters"].flight.map(targets)
    result["parameters"]["fit_status"] = result["parameters"].flight.map(statuses)
    result["assumptions"]["volume_anchor"] = "external_primary_seats_calibrates_expectation_not_an_independent_prediction"
    result["primary_demand_seats"] = targets
    return result


def _conditional_group_counts(intensities, total_seats, rng):
    """Draw compound-Poisson counts conditional on the exact weighted total.

    If nu_q is the independent Poisson intensity of groups of size q, coefficient
    h_n of exp(sum_q nu_q*z**q) obeys n*h_n=sum_q q*nu_q*h_(n-q).
    Repeated size-biased draws q*nu_q*h_(n-q)/(n*h_n) give the exact conditional
    unordered count distribution. No arbitrary final singleton is inserted.
    """
    total_seats = int(total_seats)
    if total_seats < 0:
        raise ValueError("The conditional seat total cannot be negative.")
    quantities = np.array(sorted(q for q, value in intensities.items() if value > 0), dtype=int)
    if total_seats == 0:
        return []
    if len(quantities) == 0:
        raise ValueError("No positive hidden-group intensities for a positive seat total.")
    rates = np.array([float(intensities[int(q)]) for q in quantities])
    # Log coefficients avoid overflow for large user-supplied volume anchors.
    log_h = np.full(total_seats+1, -np.inf)
    log_h[0] = 0.0
    for n in range(1, total_seats+1):
        possible = quantities <= n
        terms = np.log(quantities[possible]*rates[possible])+log_h[n-quantities[possible]]
        finite = np.isfinite(terms)
        if finite.any():
            largest = terms[finite].max()
            log_h[n] = largest+np.log(np.exp(terms[finite]-largest).sum())-np.log(n)
    if not np.isfinite(log_h[total_seats]):
        raise ValueError(f"Exact total {total_seats} is unreachable with supported group sizes {quantities.tolist()}.")
    answer, remaining = [], total_seats
    while remaining:
        possible = quantities <= remaining
        candidates = quantities[possible]
        scores = np.log(candidates*rates[possible])+log_h[remaining-candidates]
        finite = np.isfinite(scores)
        candidates, scores = candidates[finite], scores[finite]
        probabilities = np.exp(scores-scores.max())
        probabilities /= probabilities.sum()
        quantity = int(rng.choice(candidates, p=probabilities))
        answer.append(quantity)
        remaining -= quantity
    rng.shuffle(answer)
    return answer


def _draw_amount(location, sigma, price, mode, rng):
    if mode == "unrestricted":
        return float(np.exp(location+sigma*rng.normal()))
    probability = _NORMAL.cdf((log(float(price))-location)/sigma)
    u = rng.random()
    if mode == "below_price":
        p = u*probability
    elif mode == "above_price":
        p = probability+u*(1-probability)
    else:
        raise ValueError("Unknown amount-conditioning mode.")
    p = float(np.clip(p, np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0)))
    amount = float(np.exp(location+sigma*_NORMAL.inv_cdf(p)))
    # Correct only floating-point endpoint roundoff, never the generated family.
    return min(amount, np.nextafter(float(price), 0)) if mode == "below_price" else max(amount, float(price))


def sample_passengers_v2(inputs, result, seed=20261109):
    """One passenger-like realization with EXACT external primary seats/flight.

    Observed transaction time/cabin/RBD/qty/price/sign stay fixed. Their IDs and
    successful amounts are model-generated and explicitly labelled. Cancellation
    identity links cannot be recovered and remain missing. Hidden group count,
    size, reason, time and amount are drawn jointly from marked CP intensities,
    conditional on the known total; stratum totals remain random.

    Availability reasons are consequences of supplied/assumed offer states, not
    proven original reasons for unobserved passengers. Hidden cabin assignment
    is a latent model label, not a recovered observed preference.
    """
    if "primary_demand_seats" not in result:
        raise ValueError("Sampling requires fit_model_v2 and its external seat totals.")
    rng = np.random.default_rng(seed)
    config = result["config"]
    intercepts = result["parameters"].set_index("flight").mu_log.to_dict()
    offsets = config.cabin_offsets or {}
    records = []
    for i, event in enumerate(inputs.bookings.sort_values(["flight", "timestamp", "source_row"]).to_dict("records")):
        success = event["commercial"] > 0
        location = (intercepts[event["flight"]]+config.beta_time*log((1+event["dtd"])/31)
                    + offsets.get(event["cabin"], 0)+config.beta_qty*log(event["qty"]))
        amount = _draw_amount(location, config.sigma, event["price"], "above_price", rng) if success else np.nan
        records.append({"timestamp": event["timestamp"], "id": f"REC-{event['flight']}-{i:05d}",
                        "flight": event["flight"], "cabin": event["cabin"], "rbd": event.get("rbd", pd.NA),
                        "qty": event["qty"], "price": event["price"], "amount": amount,
                        "outcome": "success" if success else "cancel", "signed_qty": event["commercial"],
                        "dtd": event["dtd"], "period": event["period"], "source_row": event["source_row"],
                        "source_file": event.get("source_file", pd.NA), "cabin_source": "observed_booking",
                        "row_source": "observed_booking_operation", "id_is_synthetic": True,
                        "amount_source": "model_conditional_given_success" if success else "not_available",
                        "reason": "observed_purchase" if success else "observed_cancel_unknown_link",
                        "reason_is_modelled": False, "cancel_link_id": pd.NA})
    marks = result["marks"]
    for flight, primary_seats in result["primary_demand_seats"].items():
        group = marks.loc[marks.flight.eq(flight)&marks.expected_hidden_groups.gt(0)].copy()
        gross = int(inputs.purchases.loc[inputs.purchases.flight.eq(flight), "qty"].sum())
        target = int(primary_seats)-gross
        intensities = group.groupby("qty").expected_hidden_groups.sum().to_dict()
        quantities = _conditional_group_counts(intensities, target, rng)
        for i, quantity in enumerate(quantities):
            options = group.loc[group.qty.eq(quantity)]
            probabilities = options.expected_hidden_groups.to_numpy(dtype=float, copy=True)
            probabilities /= probabilities.sum()
            chosen = options.iloc[int(rng.choice(len(options), p=probabilities))]
            span = (chosen.end-chosen.start).total_seconds()
            timestamp = chosen.start+pd.Timedelta(seconds=float(rng.random()*span))
            mode = "unrestricted" if chosen.reason == "availability" else "below_price" if chosen.reason == "budget" else "above_price"
            amount = _draw_amount(float(chosen.wtp_log_location), config.sigma, chosen.price, mode, rng)
            records.append({"timestamp": timestamp, "id": f"SYN-{seed}-{flight}-{i:05d}",
                            "flight": flight, "cabin": pd.NA, "model_preferred_cabin": chosen.cabin,
                            "cabin_source": "not_observed_latent_model_label", "rbd": pd.NA, "qty": quantity,
                            "price": np.nan, "offered_price": chosen.price, "amount": amount,
                            "outcome": "waste", "signed_qty": 0, "dtd": int(chosen.dtd),
                            "period": chosen.period, "source_row": np.nan,
                            "source_file": pd.NA,
                            "row_source": "model_generated_hidden_group", "id_is_synthetic": True,
                            "amount_source": f"model_conditional_{chosen.reason}",
                            "reason": chosen.reason, "reason_is_modelled": True, "cancel_link_id": pd.NA})
    answer = pd.DataFrame(records).sort_values(["flight", "timestamp", "id"]).reset_index(drop=True)
    for flight, primary_seats in result["primary_demand_seats"].items():
        total = int(answer.loc[answer.flight.eq(flight)&answer.outcome.isin(["success", "waste"]), "qty"].sum())
        if total != int(primary_seats):
            raise AssertionError("Conditional passenger generation violated the primary-seat total.")
    answer.attrs["conditioning"] = "marked_compound_poisson_given_exact_primary_seats_minus_observed_gross_sales"
    answer.attrs["assumptions"] = result["assumptions"]
    return answer
