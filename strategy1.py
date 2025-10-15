"""
CTC Derivatives Trading Game — Remaining-Sum Adaptive Skew (RSAS)

Fresh strategy that prices AND trades the **sum of the remaining rolls**.

Core ideas
----------
1) Estimate per-roll mean/std empirically from (training + current) rolls.
2) Fair value for each subround = mu_hat * R, where R = remaining dice.
3) Futures = primary PnL engine:
   - Vol-aware spreads from sigma_hat * sqrt(R)
   - One-sided, signal-driven quoting (tight only on informed side) when |z| is large
   - Inventory skew + auto-widen; late flatten
4) Options = selective:
   - Quote ONLY one near-ATM strike aligned with signal
   - Stop options late in the round when little optionality remains

No SciPy; closed-form normal moments via math.erf.
"""

from autograder.sdk.strategy_interface import AbstractTradingStrategy
from typing import Any, Dict, Tuple, List, Optional
import numpy as np
import math


class MyTradingStrategy(AbstractTradingStrategy):
    # ===================== CONFIG =====================

    # Futures
    TRADE_FUTURES: bool = True
    FUT_C_SPREAD: float = 0.06         # spread = FUT_C_SPREAD * sigma_R (scaled by confidence)
    FUT_MIN_SPREAD: float = 5.0

    # Options (selective; only one ATM strike aligned with signal)
    TRADE_OPTIONS: bool = True
    OPT_SPREAD_PCT: float = 0.08       # as % of fair option value
    OPT_MIN_SPREAD: float = 1.0
    OPT_MIN_FAIR: float = 0.5
    LATE_OPTION_CUTOFF: int = 2000     # stop options when R < this

    # Signal thresholds (z-score on new subround mean vs prev mean)
    Z_STRONG: float = 1.5              # one-sided futures
    Z_WEAK: float = 1.0                # enables option side selection

    # Inventory policy
    INV_SOFT_LIMIT: int = 4
    INV_HARD_LIMIT: int = 8
    INV_WIDEN_PCT: float = 0.08        # widen factor per unit over soft limit
    INV_MID_SKEW_PCT: float = 0.015    # shift mid by % * fair * position

    # Late-round policy
    FLATTEN_CUTOFF: int = 500          # start aggressively flattening when R < this

    # Misc
    MIN_BID: float = 0.1               # engine requires positive bids
    VERY_LARGE: float = 1e18           # for one-sided markets
    USE_RECENCY_WEIGHTING: bool = True # dice distribution fixed within round; keep small weight
    RECENT_WEIGHT_BASE: float = 0.20   # modest recency weighting

    # ===================== STATE =====================

    def __init__(self) -> None:
        self.team_name: str = "Unknown"
        self.dice_sides: Optional[int] = None

        # drift signal memory
        self.prev_seen_count: int = 0
        self.prev_mean_all: Optional[float] = None
        self.last_subround: int = 0

    # ===================== LIFECYCLE =====================

    def on_game_start(self, config: Dict[str, Any]) -> None:
        self.team_name = config.get("team_name", "Unknown")
        self.dice_sides = config.get("dice_sides", None)
        self.prev_seen_count = 0
        self.prev_mean_all = None
        self.last_subround = 0
        print(f"[{self.team_name}] RSAS started. Dice sides: {self.dice_sides}")

    def on_round_end(self, result: Dict[str, Any]) -> None:
        pnl = result.get("your_pnl", result.get("pnl", 0.0))
        print(f"[{self.team_name}] Round end PnL: {pnl:.2f}")
        # reset per-round signal memory
        self.prev_seen_count = 0
        self.prev_mean_all = None
        self.last_subround = 0

    def on_game_end(self, summary: Dict[str, Any]) -> None:
        total_pnl = summary.get("total_pnl", 0.0)
        final_score = summary.get("final_score", 0.0)
        print(f"[{self.team_name}] Game done. Total PnL: {total_pnl:.2f}, Score: {final_score:.2f}")

    # ===================== MAIN =====================

    def make_market(
        self,
        *,
        marketplace: Any,
        training_rolls: Any,
        my_trades: Any,
        current_rolls: Any,
        round_info: Any
    ) -> Dict[str, Tuple[float, float]]:
        # Round config (defaults align with spec)
        num_sub = int(round_info.get("num_sub_rounds", round_info.get("num_sub_rounds", 10)))
        dice_per_sub = int(round_info.get("dice_per_subround", 2000))
        total_dice = num_sub * dice_per_sub

        # Observations
        tr = np.asarray(training_rolls, dtype=float)
        cr = np.asarray(current_rolls, dtype=float)
        all_rolls = np.concatenate([tr, cr]) if tr.size + cr.size > 0 else np.array([], dtype=float)

        # Remaining dice R
        R = max(total_dice - cr.size, 0)

        # Per-roll mean/std (empirical)
        if all_rolls.size > 1:
            mean_all = float(np.mean(all_rolls))
            std_all = float(np.std(all_rolls, ddof=1))
        elif all_rolls.size == 1:
            mean_all = float(all_rolls[0])
            std_all = max(abs(mean_all) / math.sqrt(12.0), 1.0)
        else:
            # fallback to theoretical center of range if dice_sides known
            ds = float(self.dice_sides or 10000)
            mean_all = (1.0 + ds) / 2.0
            std_all = math.sqrt((ds * ds - 1.0) / 12.0)

        # Optional mild recency weighting (kept small; dice dist fixed within round)
        if self.USE_RECENCY_WEIGHTING and cr.size > 0 and tr.size > 0:
            w = self.RECENT_WEIGHT_BASE
            recent_mean = float(np.mean(cr[-dice_per_sub:])) if cr.size >= dice_per_sub else float(np.mean(cr))
            mean_hat = w * recent_mean + (1.0 - w) * mean_all
        else:
            mean_hat = mean_all

        std_hat = max(std_all, 1e-9)  # guard

        # Fair & uncertainty for the REMAINING sum
        fair_remaining = mean_hat * R
        sigma_R = std_hat * math.sqrt(max(R, 1))

        # Build drift z-score (new subround vs prior mean)
        new_count = cr.size - self.prev_seen_count
        if new_count > 0:
            new_chunk = cr[-new_count:]
            chunk_mean = float(np.mean(new_chunk))
            baseline = self.prev_mean_all if self.prev_mean_all is not None else mean_all
            denom = max(std_hat / math.sqrt(max(new_count, 1)), 1e-9)
            z = (chunk_mean - baseline) / denom
        else:
            z = 0.0

        # Parse instruments
        products = marketplace.get_products()
        futures, calls, puts = self._split_products(products)

        # Choose one ATM strike (closest to fair)
        atm_call = self._closest_strike(calls, fair_remaining)
        atm_put = self._closest_strike(puts, fair_remaining)

        quotes: Dict[str, Tuple[float, float]] = {}
        round_id = round_info.get("round_id", 0)

        # === Futures (primary) ===
        if self.TRADE_FUTURES:
            for fut in futures:
                pid = self._pid(fut)
                pos = self._get_position(my_trades, pid)
                q = self._quote_future(
                    fair=fair_remaining,
                    sigma_R=sigma_R,
                    position=pos,
                    strong_buy=(z > self.Z_STRONG),
                    strong_sell=(z < -self.Z_STRONG)
                )
                if q is not None:
                    quotes[pid] = q

        # === Options (selective, aligned with signal, not late) ===
        if self.TRADE_OPTIONS and R >= self.LATE_OPTION_CUTOFF:
            if z > self.Z_WEAK and atm_put is not None:
                # Bullish signal -> prefer short puts / tighter quotes on puts
                pid, strike = atm_put
                pos = self._get_position(my_trades, pid)
                q = self._quote_option(
                    option_type="P",
                    strike=strike,
                    fair_remaining=fair_remaining,
                    sigma_R=sigma_R,
                    position=pos
                )
                if q is not None:
                    quotes[pid] = q
            elif z < -self.Z_WEAK and atm_call is not None:
                # Bearish signal -> prefer short calls
                pid, strike = atm_call
                pos = self._get_position(my_trades, pid)
                q = self._quote_option(
                    option_type="C",
                    strike=strike,
                    fair_remaining=fair_remaining,
                    sigma_R=sigma_R,
                    position=pos
                )
                if q is not None:
                    quotes[pid] = q

        # === Late-round flatten ===
        if R < self.FLATTEN_CUTOFF and quotes:
            for pid in list(quotes.keys()):
                pos = self._get_position(my_trades, pid)
                if abs(pos) == 0:
                    continue
                b, a = quotes[pid]
                widen = 1.5
                if pos > 0:   # long -> encourage selling
                    quotes[pid] = (max(self.MIN_BID, b / widen), a / widen)
                else:         # short -> encourage buying back
                    quotes[pid] = (b * widen, a * widen)

        # update drift memory
        self.prev_seen_count = cr.size
        self.prev_mean_all = mean_all
        self.last_subround = int(round_info.get("current_sub_round", self.last_subround))

        return quotes

    # ===================== QUOTING =====================

    def _quote_future(
        self,
        *,
        fair: float,
        sigma_R: float,
        position: float,
        strong_buy: bool,
        strong_sell: bool
    ) -> Optional[Tuple[float, float]]:
        if abs(position) > self.INV_HARD_LIMIT:
            return None

        # Base vol-aware spread with confidence squeeze
        spread = max(self.FUT_MIN_SPREAD, self.FUT_C_SPREAD * sigma_R)

        # Inventory impact
        inv_excess = max(0.0, abs(position) - self.INV_SOFT_LIMIT)
        spread *= (1.0 + self.INV_WIDEN_PCT * inv_excess)

        # Mid skew by inventory
        mid = fair - (self.INV_MID_SKEW_PCT * fair * position)

        # One-sided quoting on strong signal
        if strong_buy and not strong_sell:
            bid = max(self.MIN_BID, mid - 0.15 * spread)      # tight bid
            ask = mid + self.VERY_LARGE                       # offside ask
        elif strong_sell and not strong_buy:
            bid = max(self.MIN_BID, mid - self.VERY_LARGE)    # offside bid
            ask = mid + 0.15 * spread                         # tight ask
        else:
            # symmetric when no strong edge
            bid = max(self.MIN_BID, mid - spread / 2.0)
            ask = mid + spread / 2.0

        if bid >= ask:
            return None
        return (bid, ask)

    def _quote_option(
        self,
        *,
        option_type: str,
        strike: float,
        fair_remaining: float,
        sigma_R: float,
        position: float
    ) -> Optional[Tuple[float, float]]:
        if abs(position) > self.INV_HARD_LIMIT:
            return None

        fair = self._european_option_normal(
            option_type=option_type,
            mean=fair_remaining,
            std=max(sigma_R, 1e-9),
            strike=strike
        )

        if fair < self.OPT_MIN_FAIR:
            return None

        spread = max(self.OPT_MIN_SPREAD, self.OPT_SPREAD_PCT * fair)

        inv_excess = max(0.0, abs(position) - self.INV_SOFT_LIMIT)
        spread *= (1.0 + self.INV_WIDEN_PCT * inv_excess)

        mid = fair - (self.INV_MID_SKEW_PCT * fair * position)
        bid = max(self.MIN_BID, mid - spread / 2.0)
        ask = mid + spread / 2.0

        if bid >= ask:
            return None
        return (bid, ask)

    # ===================== PRICING HELPERS =====================

    def _european_option_normal(self, *, option_type: str, mean: float, std: float, strike: float) -> float:
        """
        E[(S-K)+] and E[(K-S)+] when S ~ N(mean, std^2).
        """
        if std < 1e-12:
            return max(0.0, mean - strike) if option_type == "C" else max(0.0, strike - mean)

        d = (mean - strike) / std
        phi = self._norm_pdf(d)
        Phi = self._norm_cdf(d)

        if option_type == "C":
            return max(0.0, (mean - strike) * Phi + std * phi)
        else:
            return max(0.0, (strike - mean) * self._norm_cdf(-d) + std * phi)

    # ===================== UTIL =====================

    def _split_products(self, products: List[Any]):
        futures, calls, puts = [], [], []
        for p in products:
            pid = self._pid(p)
            parts = pid.split(",")
            if len(parts) < 2:
                continue
            t = parts[1]
            if t == "F":
                futures.append(p)
            elif t == "C":
                # strike stored at index 2 in "S,C,STRIKE,EXPIRY"
                strike = float(parts[2]) if len(parts) > 2 else 0.0
                calls.append((pid, strike, p))
            elif t == "P":
                strike = float(parts[2]) if len(parts) > 2 else 0.0
                puts.append((pid, strike, p))
        # Return simplified structures: futures list, and (pid,strike) maps for options
        call_simple = [(pid, strike) for (pid, strike, _) in calls]
        put_simple = [(pid, strike) for (pid, strike, _) in puts]
        return futures, call_simple, put_simple

    def _closest_strike(self, opts: List[Tuple[str, float]], fair: float) -> Optional[Tuple[str, float]]:
        if not opts:
            return None
        pid, strike = min(opts, key=lambda x: abs(x[1] - fair))
        return (pid, strike)

    @staticmethod
    def _pid(prod: Any) -> str:
        return getattr(prod, "product_id", None) or getattr(prod, "id", None)

    @staticmethod
    def _get_position(my_trades: Any, product_id: str) -> float:
        pos_obj = my_trades.get_position(product_id)
        if pos_obj is None:
            return 0.0
        return float(getattr(pos_obj, "position", 0.0))

    @staticmethod
    def _norm_pdf(x: float) -> float:
        return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

    @staticmethod
    def _norm_cdf(x: float) -> float:
        # numerically stable normal CDF using erf
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
