import os
import time
import json
import argparse
import random
from typing import Any, Optional
import requests


# ----------------------------
# Helpers
# ----------------------------
def build_headers(api_key: str) -> dict[str, str]:
    return {"X-API-Key": api_key, "Content-Type": "application/json"}


def _raise_for_api_error(resp: requests.Response) -> None:
    if 200 <= resp.status_code < 300:
        return
    try:
        data = resp.json()
        detail = data.get("detail") if isinstance(data, dict) else None
    except Exception:
        detail = None
    msg = f"HTTP {resp.status_code}"
    if detail:
        msg += f": {detail}"
    raise RuntimeError(msg)


def api_get(base_url: str, path: str, api_key: str, params: Optional[dict[str, Any]] = None) -> Any:
    url = f"{base_url}{path}"
    resp = requests.get(url, headers=build_headers(api_key), params=params, timeout=15)
    _raise_for_api_error(resp)
    return resp.json()


def api_post(base_url: str, path: str, api_key: str, body: dict[str, Any]) -> Any:
    url = f"{base_url}{path}"
    resp = requests.post(url, headers=build_headers(api_key), data=json.dumps(body), timeout=10)
    if not (200 <= resp.status_code < 300):
        try:
            err = resp.json().get("detail", "")
        except Exception:
            err = resp.text
        raise RuntimeError(f"HTTP {resp.status_code}: {err}")
    return resp.json()


def place_order(api_url: str, api_key: str, order: dict[str, Any]) -> None:
    """Send a single order to the API."""
    try:
        res = api_post(api_url, "/api/v1/orders", api_key, order)
        print(
            f"[OK] {order['side'].upper():4} {order['symbol']:5} "
            f"{order['quantity']:>3} @ {order.get('price', 'MKT')} ({order['order_type']})"
        )
    except Exception as e:
        print(f"[ERR] Failed order for {order['symbol']}: {e}")


def get_order_book(api_url: str, api_key: str, symbol: str) -> dict[str, Any]:
    """Get the order book for a symbol."""
    return api_get(api_url, f"/api/v1/orderbook/{symbol}", api_key)


def get_positions(api_url: str, api_key: str) -> dict[str, Any]:
    """Get current positions."""
    return api_get(api_url, "/api/v1/positions", api_key)

# ----------------------------
# Cancel Helpers
# ----------------------------
def cancel_order(api_url: str, api_key: str, order_id: str) -> None:
    """Cancel a specific open order by ID."""
    url = f"{api_url}/api/v1/orders/{order_id}"
    resp = requests.delete(url, headers=build_headers(api_key), timeout=10)
    if 200 <= resp.status_code < 300:
        print(f"[CANCEL] Order {order_id} cancelled successfully.")
    else:
        print(f"[CANCEL ERR] Failed to cancel order {order_id}: {resp.text}")


def cancel_all_orders(api_url: str, api_key: str) -> None:
    """Cancel all open orders."""
    url = f"{api_url}/api/v1/orders/all"
    resp = requests.delete(url, headers=build_headers(api_key), timeout=10)
    if 200 <= resp.status_code < 300:
        print("[CANCEL] All open orders cancelled successfully.")
    else:
        print(f"[CANCEL ERR] Failed to cancel all orders: {resp.text}")

def cancel_bad_orders(api_url: str, api_key: str, symbol: str, fair_value: float, threshold: float = 0.3):
    """
    Cancel any open orders for a given symbol that deviate too far from current fair value.
    """
    data = api_get(api_url, "/api/v1/orders/open", api_key)
    for o in data.get("orders", []):
        if o["symbol"] != symbol:
            continue

        status = o.get("status", "").lower()
        qty = o.get("quantity", 0)
        price = o.get("price")

        print(f"   ↳ Order ID={o['order_id']} | status={status} | qty={qty}")

        # ⚠️ Skip partially filled orders
        if "partial" in status:
            continue

        if abs(price - fair_value) > threshold:
            print(f"[CANCEL] {symbol} order @ {price} deviates from fair={fair_value:.2f}, cancelling.")
            cancel_order(api_url, api_key, o["order_id"])
# ----------------------------cancel_order
# Position Management
# ----------------------------
POSITION_LIMIT = 500  # Hard cap per symbol

def can_place_order(symbol: str, side: str, qty: int, positions: dict[str, int]) -> bool:
    """Check if this order would exceed ±POSITION_LIMIT."""
    curr = positions.get(symbol, 0)
    next_pos = curr + qty if side.lower() == "buy" else curr - qty
    if abs(next_pos) > POSITION_LIMIT:
        print(
            f"[LIMIT] Skip {side.upper()} {qty} {symbol}: "
            f"current={curr}, next={next_pos} exceeds ±{POSITION_LIMIT}"
        )
        return False
    return True


# ----------------------------
# Market-making logic
# ----------------------------
def generate_fair_values(symbols: list[str]) -> dict[str, float]:
    fair = {}
    for s in symbols:
        fair[s] = round(random.uniform(90, 250), 2)
    return fair


def update_fair_values(fair: dict[str, float], drift_std: float = 0.5) -> None:
    for s in fair:
        fair[s] += random.gauss(0, drift_std)
        fair[s] = round(fair[s], 2)


def make_bid_ask_orders(symbol: str, fair_value: float) -> list[dict[str, Any]]:
    spread = random.uniform(0.1, 0.6)
    qty = random.randint(25, 50)

    bid_px = round(fair_value - spread / 2, 2)
    ask_px = round(fair_value + spread / 2, 2)

    return [
        {"symbol": symbol, "side": "buy", "order_type": "limit", "quantity": qty, "price": bid_px},
        {"symbol": symbol, "side": "sell", "order_type": "limit", "quantity": qty, "price": ask_px},
    ]


# ----------------------------
# Arbitrage logic
# ----------------------------
def get_best_bid_ask(order_book: dict[str, Any]) -> tuple[Optional[float], Optional[float]]:
    bids = order_book.get("bids", [])
    asks = order_book.get("asks", [])
    best_bid = float(bids[0]["price"]) if bids else None
    best_ask = float(asks[0]["price"]) if asks else None
    return best_bid, best_ask


def get_mid_price(order_book: dict[str, Any]) -> Optional[float]:
    best_bid, best_ask = get_best_bid_ask(order_book)
    if best_bid is not None and best_ask is not None:
        return (best_bid + best_ask) / 2.0
    elif best_bid is not None:
        return best_bid
    elif best_ask is not None:
        return best_ask
    return None


class ArbitrageTracker:
    def __init__(self):
        self.in_position = False
        self.position_type = None
        self.entry_spread = None

    def reset(self):
        self.in_position = False
        self.position_type = None
        self.entry_spread = None


def execute_arbitrage(
    api_url: str,
    api_key: str,
    etf_book: dict[str, Any],
    aaa_book: dict[str, Any],
    bbb_book: dict[str, Any],
    ccc_book: dict[str, Any],
    tracker: ArbitrageTracker,
    positions: dict[str, int],
    entry_threshold: float = 0.50,
    exit_threshold: float = 0.10,
    max_position: int = 100
) -> None:
    etf_mid = get_mid_price(etf_book)
    aaa_mid = get_mid_price(aaa_book)
    bbb_mid = get_mid_price(bbb_book)
    ccc_mid = get_mid_price(ccc_book)

    if None in [etf_mid, aaa_mid, bbb_mid, ccc_mid]:
        print("[ARB] Missing price data, skipping...")
        return

    basket_value = aaa_mid + bbb_mid + ccc_mid
    spread = etf_mid - basket_value
    spread_pct = (spread / basket_value) * 100
    print(f"[ARB] ETF: ${etf_mid:.2f} | Basket: ${basket_value:.2f} | Spread: ${spread:.2f} ({spread_pct:.2f}%)")

    etf_pos = positions.get("ETF", 0)
    aaa_pos = positions.get("AAA", 0)
    bbb_pos = positions.get("BBB", 0)
    ccc_pos = positions.get("CCC", 0)

    # EXIT LOGIC
    if tracker.in_position:
        
        should_exit = False
        for sym in ["ETF", "AAA", "BBB", "CCC"]:
            fair_value = get_mid_price(locals()[f"{sym.lower()}_book"])
            if fair_value:
                cancel_bad_orders(api_url, api_key, sym, fair_value, threshold=0.3)
        # Exit SHORT ETF
        if tracker.position_type == "short_etf" and spread <= exit_threshold:
            should_exit = True
            print(f"[ARB] Spread converged (${spread:.2f}), exiting SHORT ETF...")
            if etf_pos < 0:
                qty = min(abs(etf_pos), max_position)
                etf_ask = get_best_bid_ask(etf_book)[1]
                if etf_ask: # and can_place_order("ETF", "buy", qty, positions)
                    place_order(api_url, api_key, {
                        "symbol": "ETF", "side": "buy", "order_type": "limit",
                        "quantity": qty, "price": etf_ask
                    })

            for sym, book, pos in [("AAA", aaa_book, aaa_pos), ("BBB", bbb_book, bbb_pos), ("CCC", ccc_book, ccc_pos)]:
                if pos > 0:
                    qty = min(pos, max_position)
                    bid = get_best_bid_ask(book)[0]
                    if bid:# and can_place_order(sym, "sell", qty, positions)
                        place_order(api_url, api_key, {
                            "symbol": sym, "side": "sell", "order_type": "limit",
                            "quantity": qty, "price": bid
                        })

        # Exit LONG ETF
        elif tracker.position_type == "long_etf" and spread >= -exit_threshold:
            should_exit = True
            print(f"[ARB] Spread converged (${spread:.2f}), exiting LONG ETF...")
            if etf_pos > 0:
                qty = min(etf_pos, max_position)
                etf_bid = get_best_bid_ask(etf_book)[0]
                if etf_bid :#and can_place_order("ETF", "sell", qty, positions)
                    place_order(api_url, api_key, {
                        "symbol": "ETF", "side": "sell", "order_type": "limit",
                        "quantity": qty, "price": etf_bid
                    })

            for sym, book, pos in [("AAA", aaa_book, aaa_pos), ("BBB", bbb_book, bbb_pos), ("CCC", ccc_book, ccc_pos)]:
                if pos < 0:
                    qty = min(abs(pos), max_position)
                    ask = get_best_bid_ask(book)[1]
                    if ask :#and can_place_order(sym, "buy", qty, positions)
                        place_order(api_url, api_key, {
                            "symbol": sym, "side": "buy", "order_type": "limit",
                            "quantity": qty, "price": ask
                        })

        if should_exit:
            print("[ARB] Position closed — cancelling any remaining open orders.")
            cancel_all_orders(api_url, api_key)
            tracker.reset()
            return

    # ENTRY LOGIC
    if not tracker.in_position:
        cancel_all_orders(api_url, api_key)
        # SHORT ETF / LONG basket
        if spread > entry_threshold:
            print(f"[ARB] ETF overpriced by ${spread:.2f}, entering SHORT ETF / LONG basket...")
            etf_bid = get_best_bid_ask(etf_book)[0]
            if etf_bid :#and can_place_order("ETF", "sell", max_position, positions)
                place_order(api_url, api_key, {
                    "symbol": "ETF", "side": "sell", "order_type": "limit",
                    "quantity": max_position, "price": etf_bid
                })
            for sym, book in [("AAA", aaa_book), ("BBB", bbb_book), ("CCC", ccc_book)]:
                ask = get_best_bid_ask(book)[1]
                if ask :#and can_place_order(sym, "buy", max_position, positions)
                    place_order(api_url, api_key, {
                        "symbol": sym, "side": "buy", "order_type": "limit",
                        "quantity": max_position, "price": ask
                    })
            tracker.in_position = True
            tracker.position_type = "short_etf"
            tracker.entry_spread = spread

        # LONG ETF / SHORT basket
        elif spread < -entry_threshold:
            print(f"[ARB] ETF underpriced by ${abs(spread):.2f}, entering LONG ETF / SHORT basket...")
            etf_ask = get_best_bid_ask(etf_book)[1]
            if etf_ask : #and can_place_order("ETF", "buy", max_position, positions)
                place_order(api_url, api_key, {
                    "symbol": "ETF", "side": "buy", "order_type": "limit",
                    "quantity": max_position, "price": etf_ask
                })
            for sym, book in [("AAA", aaa_book), ("BBB", bbb_book), ("CCC", ccc_book)]:
                bid = get_best_bid_ask(book)[0]
                if bid: #and can_place_order(sym, "sell", max_position, positions)
                    place_order(api_url, api_key, {
                        "symbol": sym, "side": "sell", "order_type": "limit",
                        "quantity": max_position, "price": bid
                    })
            tracker.in_position = True
            tracker.position_type = "long_etf"
            tracker.entry_spread = spread


# ----------------------------
# Main loops
# ----------------------------
def arbitrage_loop(api_url: str, api_key: str, entry_threshold: float, exit_threshold: float, max_position: int, loop: bool):
    tracker = ArbitrageTracker()
    print(f"[ARB] Starting arbitrage bot (entry: ${entry_threshold}, exit: ${exit_threshold})")

    while True:
        try:
            etf_book = get_order_book(api_url, api_key, "ETF")
            aaa_book = get_order_book(api_url, api_key, "AAA")
            bbb_book = get_order_book(api_url, api_key, "BBB")
            ccc_book = get_order_book(api_url, api_key, "CCC")

            pos_data = get_positions(api_url, api_key)
            positions = {pos["symbol"]: pos["quantity"] for pos in pos_data.get("positions", [])}

            execute_arbitrage(api_url, api_key, etf_book, aaa_book, bbb_book, ccc_book,
                              tracker, positions, entry_threshold, exit_threshold, max_position)

        except Exception as e:
            print(f"[ARB] Error in arbitrage loop: {e}")

        if not loop:
            break
        time.sleep(1)


def market_making_loop(api_url: str, api_key: str, symbols: list[str], loop: bool = True):
    fair = generate_fair_values(symbols)
    print("Initial fair values:", fair)

    while True:
        try:
            pos_data = get_positions(api_url, api_key)
            positions = {pos["symbol"]: pos["quantity"] for pos in pos_data.get("positions", [])}

            update_fair_values(fair, drift_std=0.3)
            for sym in symbols:
                fair_value = fair[sym]
                orders = make_bid_ask_orders(sym, fair_value)
                for o in orders:
                    # if not can_place_order(o["symbol"], o["side"], o["quantity"], positions):
                    #     continue
                    place_order(api_url, api_key, o)
                print(f"[{sym}] fair={fair_value:.2f}\n")
                time.sleep(0.5)

        except Exception as e:
            print(f"[MM] Error: {e}")

        if not loop:
            break
        time.sleep(1)


# ----------------------------
# Entry point
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Automated Trading Bot for CTC API")
    parser.add_argument("--api-url", default=os.environ.get("CTC_API_URL", "http://localhost:8000"))
    parser.add_argument("--api-key", default=os.environ.get("CTC_API_KEY") or os.environ.get("X_API_KEY"))
    parser.add_argument("--mode", default="arbitrage", choices=["arbitrage", "market_making"])
    parser.add_argument("--symbols", default="AAA,BBB,CCC,ETF")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--entry-threshold", type=float, default=0.50)
    parser.add_argument("--exit-threshold", type=float, default=0.10)
    parser.add_argument("--max-position", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    api_key = args.api_key or input("Enter API key: ").strip()
    if not api_key:
        print("API key required.")
        return 1

    if args.mode == "arbitrage":
        arbitrage_loop(args.api_url, api_key, args.entry_threshold, args.exit_threshold, args.max_position, args.loop)
    else:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        market_making_loop(args.api_url, api_key, symbols, loop=args.loop)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
