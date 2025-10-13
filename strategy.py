from autograder.sdk.strategy_interface import AbstractTradingStrategy
from typing import Any, Dict, Tuple, List
import numpy as np
import math

class MyTradingStrategy(AbstractTradingStrategy):
    """
    Clean market-making strategy for CTC Derivatives.

    Core approach:
    1. Use training + current rolls to estimate mean and std of dice distribution
    2. Price futures at expected final sum
    3. Price options using normal approximation (Black-Scholes style)
    4. Focus on near-ATM strikes for best liquidity
    5. Adjust spreads based on uncertainty (remaining dice) and inventory
    """

    # ============ CONFIGURATION ============

    # Futures settings
    FUT_BASE_SPREAD = 0.10      # 10% of remaining std
    FUT_MIN_SPREAD = 5.0        # Minimum spread on futures

    # Options settings
    OPT_BASE_SPREAD = 0.15      # 15% of option fair value
    OPT_MIN_SPREAD = 2.0        # Minimum spread on options
    OPT_VOL_ADJUST = True       # Adjust spreads based on realized volatility
    OPT_TAIL_ADJUST = True      # Adjust spreads for fat-tailed distributions

    # Strike selection
    NUM_STRIKES_PER_SIDE = 3    # Quote 3 calls and 3 puts near ATM

    # Inventory management
    INV_SKEW_FACTOR = 0.01      # Shift mid by 1% per unit of inventory
    INV_WIDEN_FACTOR = 0.05     # Widen spread by 5% per unit beyond limit
    INV_SOFT_LIMIT = 5          # Start widening spreads beyond this
    INV_HARD_LIMIT = 10         # Stop quoting beyond this

    # Risk controls
    MIN_BID = 0.1               # Engine requires positive bids
    MAX_SPREAD_RATIO = 5.0      # Don't quote if spread > 5x fair value

    def __init__(self):
        self.team_name = "Unknown"
        self.dice_sides = None
        self.dist_stats = {}  # Store distribution statistics

    # ============ LIFECYCLE METHODS ============

    def on_game_start(self, config: Dict[str, Any]) -> None:
        self.team_name = config.get("team_name", "Unknown")
        self.dice_sides = config.get("dice_sides", None)
        print(f"[{self.team_name}] Game started. Dice sides: {self.dice_sides}")

    def on_round_end(self, result: Dict[str, Any]) -> None:
        pnl = result.get("pnl", 0.0)
        print(f"[{self.team_name}] Round ended. PnL: {pnl:.2f}")

    def on_game_end(self, summary: Dict[str, Any]) -> None:
        total_pnl = summary.get("total_pnl", 0.0)
        final_score = summary.get("final_score", 0.0)
        print(f"[{self.team_name}] Game ended. Total PnL: {total_pnl:.2f}, Score: {final_score:.2f}")

    # ============ MAIN TRADING LOGIC ============

    def make_market(
        self,
        *,
        marketplace: Any,
        training_rolls: Any,
        my_trades: Any,
        current_rolls: Any,
        round_info: Any
    ) -> Dict[str, Tuple[float, float]]:
        """
        Generate bid-ask quotes for all products.
        """

        # Get configuration
        num_subrounds = int(round_info.get("num_sub_rounds", 10))
        dice_per_subround = int(round_info.get("dice_per_subround", 2000))
        total_dice = num_subrounds * dice_per_subround

        # Combine all observed rolls to estimate distribution
        all_rolls = list(training_rolls) + list(current_rolls)

        if len(all_rolls) == 0:
            # Fallback: assume uniform distribution
            per_roll_mean = (1 + (self.dice_sides or 10000)) / 2.0
            per_roll_std = ((self.dice_sides or 10000)**2 - 1) / 12.0
            per_roll_std = math.sqrt(per_roll_std)
            vol_multiplier = 1.0
        else:
            per_roll_mean = float(np.mean(all_rolls))
            per_roll_std = float(np.std(all_rolls, ddof=1)) if len(all_rolls) > 1 else 1.0

            # Analyze distribution shape for better pricing
            vol_multiplier = self._analyze_distribution(all_rolls, per_roll_mean, per_roll_std)

        # Calculate revealed and remaining dice
        num_revealed = len(current_rolls)
        num_remaining = max(0, total_dice - num_revealed)

        # Expected final sum
        sum_revealed = float(np.sum(current_rolls)) if num_revealed > 0 else 0.0
        expected_final_sum = sum_revealed + per_roll_mean * num_remaining

        # Remaining uncertainty (standard deviation of final sum)
        std_remaining = per_roll_std * math.sqrt(max(1, num_remaining))

        # Adjust for distribution characteristics
        if len(all_rolls) > 0:
            std_remaining_adjusted = std_remaining * vol_multiplier
        else:
            std_remaining_adjusted = std_remaining

        # Get all products and categorize them
        products = marketplace.get_products()
        futures = []
        calls = []
        puts = []

        for product in products:
            pid = getattr(product, "product_id", None) or getattr(product, "id", None)
            parts = pid.split(",")
            product_type = parts[1]

            if product_type == "F":
                futures.append(product)
            elif product_type == "C":
                strike = float(parts[2])
                calls.append((strike, product))
            elif product_type == "P":
                strike = float(parts[2])
                puts.append((strike, product))

        # Select near-ATM strikes
        calls.sort(key=lambda x: abs(x[0] - expected_final_sum))
        puts.sort(key=lambda x: abs(x[0] - expected_final_sum))

        selected_calls = calls[:self.NUM_STRIKES_PER_SIDE]
        selected_puts = puts[:self.NUM_STRIKES_PER_SIDE]

        # Build quotes
        quotes = {}

        # Quote futures (use unadjusted std for futures)
        for product in futures:
            quote = self._quote_future(
                product=product,
                fair_value=expected_final_sum,
                std_remaining=std_remaining,
                my_trades=my_trades
            )
            if quote:
                pid = getattr(product, "product_id", None) or getattr(product, "id", None)
                quotes[pid] = quote

        # Quote options (use adjusted std for options to account for vol characteristics)
        for strike, product in selected_calls + selected_puts:
            pid = getattr(product, "product_id", None) or getattr(product, "id", None)
            parts = pid.split(",")
            option_type = parts[1]

            quote = self._quote_option(
                product=product,
                option_type=option_type,
                strike=strike,
                expected_final_sum=expected_final_sum,
                std_remaining=std_remaining_adjusted,  # Use adjusted std
                my_trades=my_trades
            )
            if quote:
                quotes[pid] = quote

        return quotes

    # ============ PRICING METHODS ============

    def _quote_future(
        self,
        product: Any,
        fair_value: float,
        std_remaining: float,
        my_trades: Any
    ) -> Tuple[float, float] | None:
        """
        Generate bid-ask quote for a future.
        """
        pid = getattr(product, "product_id", None) or getattr(product, "id", None)
        position = self._get_position(my_trades, pid)

        # Check hard inventory limit
        if abs(position) > self.INV_HARD_LIMIT:
            return None

        # Base spread scales with remaining uncertainty
        base_spread = max(self.FUT_MIN_SPREAD, self.FUT_BASE_SPREAD * std_remaining)

        # Inventory adjustments
        mid_skew = fair_value * self.INV_SKEW_FACTOR * position

        inv_excess = max(0, abs(position) - self.INV_SOFT_LIMIT)
        spread_widen = 1.0 + self.INV_WIDEN_FACTOR * inv_excess
        spread = base_spread * spread_widen

        # Calculate bid and ask
        mid = fair_value - mid_skew
        bid = max(self.MIN_BID, mid - spread / 2)
        ask = mid + spread / 2

        # Validate
        if bid >= ask:
            return None

        return (bid, ask)

    def _quote_option(
        self,
        product: Any,
        option_type: str,
        strike: float,
        expected_final_sum: float,
        std_remaining: float,
        my_trades: Any
    ) -> Tuple[float, float] | None:
        """
        Generate bid-ask quote for an option.
        """
        pid = getattr(product, "product_id", None) or getattr(product, "id", None)
        position = self._get_position(my_trades, pid)

        # Check hard inventory limit
        if abs(position) > self.INV_HARD_LIMIT:
            return None

        # Price option using normal approximation
        fair_value = self._price_option_normal(
            option_type=option_type,
            strike=strike,
            expected_value=expected_final_sum,
            std=std_remaining
        )

        # Base spread as percentage of fair value
        base_spread = max(self.OPT_MIN_SPREAD, self.OPT_BASE_SPREAD * fair_value)

        # Don't quote if spread is too wide relative to fair value
        if fair_value > 0 and base_spread / fair_value > self.MAX_SPREAD_RATIO:
            return None

        # Inventory adjustments
        mid_skew = fair_value * self.INV_SKEW_FACTOR * position

        inv_excess = max(0, abs(position) - self.INV_SOFT_LIMIT)
        spread_widen = 1.0 + self.INV_WIDEN_FACTOR * inv_excess
        spread = base_spread * spread_widen

        # Calculate bid and ask
        mid = fair_value - mid_skew
        bid = max(self.MIN_BID, mid - spread / 2)
        ask = mid + spread / 2

        # Validate
        if bid >= ask:
            return None

        return (bid, ask)

    def _price_option_normal(
        self,
        option_type: str,
        strike: float,
        expected_value: float,
        std: float
    ) -> float:
        """
        Price European option assuming final sum ~ N(expected_value, std^2).

        For a call: E[(S - K)+] = (� - K) * �(d) + � * �(d)
        For a put:  E[(K - S)+] = (K - �) * �(-d) + � * �(d)

        where d = (� - K) / �
        """
        if std < 1e-9:
            # No uncertainty left, use intrinsic value
            if option_type == "C":
                return max(0.0, expected_value - strike)
            else:
                return max(0.0, strike - expected_value)

        d = (expected_value - strike) / std
        phi_d = self._normal_pdf(d)
        Phi_d = self._normal_cdf(d)

        if option_type == "C":
            value = (expected_value - strike) * Phi_d + std * phi_d
        else:  # Put
            value = (strike - expected_value) * self._normal_cdf(-d) + std * phi_d

        return max(0.0, value)

    # ============ HELPER METHODS ============

    def _analyze_distribution(self, rolls: list, mean: float, std: float) -> float:
        """
        Analyze distribution characteristics to adjust volatility pricing.

        Returns a multiplier for standard deviation:
        - > 1.0 if distribution has fat tails or high kurtosis (widen option spreads)
        - < 1.0 if distribution is tighter than normal (tighten option spreads)
        - = 1.0 for normal-like distributions

        This captures what you asked about: {3,4} vs {1,6} dice have same mean
        but very different risk profiles!
        """
        if len(rolls) < 10 or std < 1e-9:
            return 1.0

        rolls_array = np.array(rolls)

        # Calculate coefficient of variation (relative volatility)
        # High CV = high volatility relative to mean
        cv = std / abs(mean) if abs(mean) > 1e-9 else 0.0

        # Calculate kurtosis (measure of tail heaviness)
        # Kurtosis = 3 for normal distribution
        # > 3 means fat tails (extreme events more likely)
        # < 3 means thin tails (less extreme events)
        if len(rolls) > 3:
            standardized = (rolls_array - mean) / std
            kurtosis = float(np.mean(standardized ** 4))
        else:
            kurtosis = 3.0

        # Calculate range relative to std (another tail measure)
        # For normal dist, ~99.7% of data is within 3 std
        data_range = float(np.max(rolls_array) - np.min(rolls_array))
        range_ratio = data_range / (6 * std) if std > 1e-9 else 1.0

        # Build multiplier based on distribution characteristics
        multiplier = 1.0

        # Adjust for kurtosis (fat tails)
        if self.OPT_TAIL_ADJUST:
            if kurtosis > 3.5:  # Fat tails
                # Widen spreads by up to 30% for very fat tails
                multiplier *= 1.0 + 0.1 * (kurtosis - 3.0)
            elif kurtosis < 2.5:  # Thin tails
                # Tighten spreads slightly
                multiplier *= 0.9

        # Adjust for realized volatility regime
        if self.OPT_VOL_ADJUST:
            if cv > 0.5:  # High relative volatility
                multiplier *= 1.1
            elif cv < 0.2:  # Low relative volatility
                multiplier *= 0.9

        # Cap the multiplier to reasonable bounds
        multiplier = max(0.7, min(1.5, multiplier))

        # Store for debugging
        self.dist_stats = {
            "mean": mean,
            "std": std,
            "cv": cv,
            "kurtosis": kurtosis,
            "range_ratio": range_ratio,
            "vol_multiplier": multiplier
        }

        return multiplier

    def _get_position(self, my_trades: Any, product_id: str) -> float:
        """Get current position for a product."""
        pos_obj = my_trades.get_position(product_id)
        if pos_obj is None:
            return 0.0
        return float(getattr(pos_obj, "position", 0.0))

    @staticmethod
    def _normal_pdf(x: float) -> float:
        """Standard normal probability density function."""
        return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

    @staticmethod
    def _normal_cdf(x: float) -> float:
        """Standard normal cumulative distribution function."""
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))