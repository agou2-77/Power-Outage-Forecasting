"""Part II: Backup-generator pre-positioning policy.

Utility has five mobile generators. Each generator restores power for up to
1,000 households at the county it is deployed to. Once deployed, the generator
stays at that county for the duration of the event. Multiple generators may
be assigned to the same county.

Goal: maximize the total customer-hours of restored power over the 48-hour
forecast window, subject to the per-generator household cap.

Formalization
-------------
Let `p[c, h]` be the predicted number of outages in county c at hour h
(for h = 1..48 covering the forecast window).
Let `G[c]` be the number of generators assigned to county c (Σ G[c] = 5).
Each generator serves up to 1,000 households, so the households restored in
county c at hour h is:  `restored[c, h] = min(p[c, h], 1000 * G[c])`.

Objective:  maximize  Σ_h Σ_c restored[c, h]   s.t.   Σ_c G[c] = 5, G[c] ≥ 0.

Because `restored` is separable, concave in G[c] (piecewise-linear with
capacity saturation at 1000 * G[c] = max_h p[c, h]), and the constraint is a
simple budget, the optimum can be found by a greedy marginal-benefit
allocation: allocate generators one-at-a-time to the county with the highest
*marginal* total restored outages.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


GENERATOR_CAPACITY = 1_000  # households per generator
N_GENERATORS = 5


@dataclass(frozen=True)
class Assignment:
    """Immutable result of a generator-placement decision."""
    counts: dict              # FIPS -> number of generators
    fips_list: list           # length 5, each entry a FIPS string
    expected_customer_hours: float
    rationale: dict           # marginal gain per step, for reporting


def total_restored(pred: np.ndarray, G_c: int, cap: int = GENERATOR_CAPACITY) -> float:
    """Sum over the horizon of min(pred[t], G_c * cap)."""
    return float(np.minimum(pred, G_c * cap).sum())


def greedy_allocation(pred: np.ndarray, locations: np.ndarray,
                      n_gen: int = N_GENERATORS,
                      cap: int = GENERATOR_CAPACITY) -> Assignment:
    """Greedy marginal-benefit allocation of generators across counties.

    pred: (L, H) predicted outages.
    locations: (L,) FIPS codes.
    Returns an Assignment with one FIPS per generator (length n_gen).
    """
    L = pred.shape[0]
    counts = np.zeros(L, dtype=np.int32)
    rationale = {"steps": []}
    # Current restored per county given current counts
    current_restored = np.zeros(L, dtype=np.float64)
    for g in range(n_gen):
        # Marginal benefit of adding one more generator to county i:
        #   new_restored - current_restored
        # where new_restored = sum min(pred[i], (counts[i]+1)*cap)
        marginal = np.empty(L, dtype=np.float64)
        for i in range(L):
            new_r = np.minimum(pred[i], (counts[i] + 1) * cap).sum()
            marginal[i] = new_r - current_restored[i]
        choice = int(np.argmax(marginal))
        counts[choice] += 1
        current_restored[choice] += marginal[choice]
        rationale["steps"].append({
            "step": g + 1,
            "chosen_fips": str(locations[choice]),
            "marginal_restored_customer_hours": float(marginal[choice]),
        })

    fips_list = []
    for i in range(L):
        fips_list.extend([str(locations[i])] * counts[i])
    counts_dict = {str(locations[i]): int(counts[i]) for i in range(L) if counts[i] > 0}
    return Assignment(
        counts=counts_dict,
        fips_list=fips_list,
        expected_customer_hours=float(current_restored.sum()),
        rationale=rationale,
    )


def top_predicted_counties(pred: np.ndarray, locations: np.ndarray,
                            k: int = 10) -> list:
    """Helper for reporting: counties ranked by total predicted outages."""
    totals = pred.sum(axis=1)
    order = np.argsort(-totals)[:k]
    return [(str(locations[i]), float(totals[i]), float(pred[i].max())) for i in order]
