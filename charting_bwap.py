import time, requests
import os
from dotenv import load_dotenv
load_dotenv()
ETHERSCAN_KEY = os.getenv("ETHERSCAN_API_KEY")  # load from env

def get_first_tx_block(addr):
    import requests
    url = "https://api.etherscan.io/v2/api"
    params = {
        "chainid": 1,
        "module": "account",
        "action": "tokentx",
        "address": addr,          # <-- address, NOT contractaddress
        "page": 1,
        "offset": 1,
        "sort": "asc",
        "apikey": ETHERSCAN_KEY,
    }
    r = requests.get(url, params=params, timeout=30).json()
    if r.get("status") != "1" or not r.get("result"):
        print("first_tx_block error:", addr, "| msg:", r.get("message"), "| result:", r.get("result"))
        return None
    return int(r["result"][0]["blockNumber"])

def getERC20Transactions(address, startBlock, endBlock, contract_address=None, max_pages=1):
    if not ETHERSCAN_KEY:
        raise RuntimeError("Missing ETHERSCAN_API_KEY environment variable")

    base_url = "https://api.etherscan.io/v2/api"
    chainid = 1

    all_txs = []
    page = 1
    offset = 10000

    while page <= max_pages:
        params = {
            "chainid": chainid,
            "module": "account",
            "action": "tokentx",
            "address": str(address),
            "page": page,
            "offset": offset,
            "startblock": int(startBlock),
            "endblock": int(endBlock),
            "sort": "asc",
            "apikey": ETHERSCAN_KEY,
        }

        if contract_address:
            params["contractaddress"] = str(contract_address)

        # retry logic
        for attempt in range(5):
            try:
                r = requests.get(base_url, params=params, timeout=60)
                r.raise_for_status()
                payload = r.json()
                break
            except requests.exceptions.RequestException:
                time.sleep(1.5 * (attempt + 1))
        else:
            raise RuntimeError(f"Etherscan request keeps timing out for blocks {startBlock}-{endBlock}")

        result = payload.get("result")
        status = payload.get("status")

        # no transactions
        if status == "0" and (result == [] or payload.get("message") == "No transactions found"):
            break

        # unexpected format
        if status != "1" or not isinstance(result, list):
            print("Etherscan error:", payload.get("message"), "| result:", result)
            break

        all_txs.extend(result)

        # last page
        if len(result) < offset:
            break

        page += 1

    return all_txs

def getAllERC20Transactions(address, maxLen, startBlock, endBlock, contract_address=None):
    all_txs = []
    cur = int(startBlock)
    endBlock = int(endBlock)

    while cur <= endBlock:
        # fetch from current cursor to the final endBlock
        txs = getERC20Transactions(
            address,
            cur,
            endBlock,
            contract_address=contract_address,
            max_pages=1
        )
        print(address, cur, endBlock, len(txs))

        if not txs:
            break

        all_txs.extend(txs)

        if maxLen is not None and len(all_txs) >= maxLen:
            print('fetched transactions (maxLen reached) for', address, len(all_txs))
            return all_txs

        # move start to the block after the last tx we got
        last_block = int(txs[-1]["blockNumber"])
        if last_block >= endBlock:
            break
        cur = last_block + 1

        time.sleep(0.2)  # be nice to the API

    print('fetched transactions for', address, len(all_txs))
    return all_txs

def clean_sandwich(x_axis, y_axis, k=1.5, neighbor_count=10):
    """
    Remove local spikes from x_axis and y_axis.

    A point y[i] is removed if it is > k * (local median of up to `neighbor_count`
    closest neighbors around i, excluding i itself).

    Args:
        x_axis (list): x values
        y_axis (list): y values (same length as x_axis)
        k (float): spike threshold multiplier (default 3)
        neighbor_count (int): number of neighbor indices to use for local median
                              (up to this many, fewer near edges)

    Returns:
        (clean_x, clean_y): lists with spikes removed
    """
    import statistics

    if len(x_axis) != len(y_axis):
        raise ValueError("x_axis and y_axis must have the same length.")

    n = len(y_axis)
    if n == 0:
        return x_axis, y_axis

    keep = [True] * n
    half_window = neighbor_count // 2  # 10 neighbors -> 5 on each side

    for i in range(n):
        # Determine local window [start, end)
        start = max(0, i - half_window)
        end = min(n, i + half_window + 1)

        # Collect neighbors excluding the point itself
        neighbors = [y_axis[j] for j in range(start, end) if j != i]

        # If no neighbors, we can't compute a local median
        if not neighbors:
            continue

        local_med = statistics.median(neighbors)

        # If local median is 0, skip spike detection to avoid weird behavior
        if local_med == 0:
            continue

        # Mark as spike if it is much larger than local median
        if y_axis[i] > k * local_med:
            keep[i] = False

    clean_x = [x for x, flag in zip(x_axis, keep) if flag]
    clean_y = [y for y, flag in zip(y_axis, keep) if flag]

    return clean_x, clean_y



def buildPoolChart(token0, token1, pair, maxLen, clean = 1):
    token0 = token0.lower()
    token1 = token1.lower()
    pair   = pair.lower()

    data = getAllERC20Transactions(pair, maxLen, get_first_tx_block(pair), endBlock=30_164_015)
    if not data:
        return [], [], [], []

    # ensure correct order
    data = sorted(data, key=lambda t: int(t["blockNumber"]))

    token0Balance = 0
    token1Balance = 0

    pool = []
    x_axis = []
    y_axis = []
    liq = []

    for tx in data:
        _token  = tx["contractAddress"].lower()
        _time   = int(tx["timeStamp"])
        _amount = int(tx["value"])
        _to     = tx["to"].lower()
        _from   = tx["from"].lower()
        _hash   = tx["hash"]
        
        #Balances
        if _token == token0 and _from == pair:
            token0Balance -= _amount
        elif _token == token1 and _from == pair:
            token1Balance -= _amount
        elif _token == token0 and _to == pair:
            token0Balance += _amount
        elif _token == token1 and _to == pair:
            token1Balance += _amount

        price = 0 if token1Balance == 0 else token0Balance / token1Balance

        pool.append([_hash, _time, price, token0Balance, token1Balance])
        x_axis.append(_time)
        y_axis.append(price)
        liq.append(token0Balance)
        
    if clean : 
        [x_axis,y_axis] = clean_sandwich(x_axis,y_axis)
    
    return x_axis, y_axis, liq, pool

def getTokenTransfers(token_contract, startBlock, endBlock, address=None, max_pages=1):
    """
    Fetch ERC20 transfer events.

    - If token_contract is not None: transfers for that token (optionally filtered by address).
    - If token_contract is None:     all token transfers for the given address in the block range.

    Retries both:
    - network / HTTP errors
    - Etherscan API errors (non-OK status), except for the "no transactions found" case.
    """
    if not ETHERSCAN_KEY:
        raise RuntimeError("Missing ETHERSCAN_API_KEY environment variable")

    if token_contract is None and address is None:
        raise ValueError("Must provide token_contract or address (or both)")

    base_url = "https://api.etherscan.io/v2/api"
    chainid = 1

    all_txs = []
    page = 1
    offset = 10000

    max_attempts = 5
    base_backoff = 1.5  # seconds

    while page <= max_pages:
        params = {
            "chainid": chainid,
            "module": "account",
            "action": "tokentx",
            "page": page,
            "offset": offset,
            "startblock": int(startBlock),
            "endblock": int(endBlock),
            "sort": "asc",
            "apikey": ETHERSCAN_KEY,
        }

        # only include contractaddress if we are filtering by a specific token
        if token_contract is not None:
            params["contractaddress"] = str(token_contract)

        # optionally filter by holder address
        if address:
            params["address"] = str(address)

        # retry logic (network + etherscan errors)
        attempt = 0
        while True:
            attempt += 1
            try:
                r = requests.get(base_url, params=params, timeout=60)
                r.raise_for_status()
                payload = r.json()
            except requests.exceptions.RequestException as e:
                if attempt >= max_attempts:
                    raise RuntimeError(
                        f"Etherscan request keeps failing "
                        f"for blocks {startBlock}-{endBlock}, page {page}: {e}"
                    )
                time.sleep(base_backoff * attempt)
                continue

            # Parse Etherscan response
            result = payload.get("result")
            status = str(payload.get("status", ""))  # etherscan usually "1" or "0"
            msg = payload.get("message", "")

            # "No transactions found" is a terminal success (no need to retry)
            if status == "0" and (result == [] or msg == "No transactions found"):
                # nothing more for this range; exit pagination entirely
                return all_txs

            # Any other non-success status from Etherscan: retry a few times
            if status != "1" or not isinstance(result, list):
                if attempt >= max_attempts:
                    raise RuntimeError(
                        f"Etherscan error for blocks {startBlock}-{endBlock}, page {page}: "
                        f"status={status}, message={msg}, result={result}"
                    )
                time.sleep(base_backoff * attempt)
                continue

            # Success case: valid list of txs
            break  # exit retry loop, process this page

        # accumulate page results
        all_txs.extend(result)

        # If we got fewer than `offset` results, we've reached the last page.
        if len(result) < offset:
            break

        page += 1

    return all_txs

def getAllTransfers(token_contract=None, maxLen=None, startBlock=None, endBlock=None, holder_address=None):
    all_txs = []

    # --- determine startBlock if not explicitly given ---
    if startBlock is None:
        starts = []

        if token_contract is not None:
            b = get_first_tx_block(token_contract)
            if b is not None:
                starts.append(b)

        if holder_address is not None:
            b = get_first_tx_block(holder_address)
            if b is not None:
                starts.append(b)

        # if we found any valid start blocks use the earliest
        if starts:
            startBlock = min(starts)
        else:
            startBlock = 0  # fallback

    # --- determine endBlock if not provided ---
    if endBlock is None:
        endBlock = 30_164_015

    cur = int(startBlock)
    endBlock = int(endBlock)

    while cur <= endBlock:
        txs = getTokenTransfers(
            token_contract,      # can be None
            cur,
            endBlock,
            address=holder_address,  # can be None
            max_pages=1
        )

        print(token_contract, cur, endBlock, len(txs))

        if not txs:
            break

        all_txs.extend(txs)

        if maxLen is not None and len(all_txs) >= maxLen:
            print("fetched transactions (maxLen reached)", len(all_txs))
            return all_txs

        last_block = int(txs[-1]["blockNumber"])
        if last_block >= endBlock:
            break

        cur = last_block + 1
        time.sleep(0.2)

    print("fetched transactions:", len(all_txs))
    return all_txs

def flatten(lst):
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result

import bisect
from collections import defaultdict

def buildHoldersDict(
    all_txs,
    x_axis,
    y_axis,
    pair_addresses,
    cex_addresses,
    router_addresses,
    avg_time_threshold=None,
    trxCountThreshold=None,
    min_snapshot_delta=0,
    track_trades=True,
    snapshot_full=False,
    verbose=False
):
    # ----------------------------------
    # 0. PREP FAST STRUCTURES
    # ----------------------------------
    pair_addresses = flatten(pair_addresses)
    pair_set = set(pair_addresses)
    cex_addresses = flatten(cex_addresses)
    cex_set = set(cex_addresses)
    router_addresses = flatten(router_addresses)
    router_set = set(router_addresses)
    ZERO = "0x0000000000000000000000000000000000000000"
    SKIP = pair_set | cex_set | router_set | {ZERO}

    # Pre-convert ALL timestamps to int ONCE
    for tx in all_txs:
        tx["timeStamp"] = int(tx["timeStamp"])
    all_txs.sort(key=lambda x: x["timeStamp"])

    # Ensure x_axis is int and sorted; assume already aligned with y_axis
    x_axis_sorted = [int(x) for x in x_axis]
    y_axis_ref = list(y_axis)
    n_prices = len(x_axis_sorted)

    if n_prices == 0:
        raise ValueError("x_axis/y_axis must not be empty")

    # ----------------------------------
    # 1. ACTIVITY ANALYSIS + ALLOWED MAP
    # ----------------------------------

    addr_ts = defaultdict(list)
    for tx in all_txs:
        t = tx["timeStamp"]
        addr_ts[tx["from"]].append(t)
        addr_ts[tx["to"]].append(t)

    allowed_map = {}
    use_filters = (avg_time_threshold is not None and trxCountThreshold is not None)

    for addr, times in addr_ts.items():
        if addr in SKIP:
            allowed_map[addr] = False
            continue

        cnt = len(times)
        if not use_filters or cnt < 2:
            allowed_map[addr] = True
            continue

        # Compute average interval between transactions
        times.sort()
        total_gap = 0
        for i in range(1, cnt):
            total_gap += times[i] - times[i - 1]
        avg_int = total_gap / (cnt - 1)

        # Exclude likely bots: high frequency + high count
        if cnt > trxCountThreshold and avg_int < avg_time_threshold:
            allowed_map[addr] = False
        else:
            allowed_map[addr] = True

    def allowed(addr):
        return allowed_map.get(addr, False)

    # ----------------------------------
    # 2. UNION–FIND + LIVE CLUSTERS
    # ----------------------------------
    parent = {}
    adjacency = defaultdict(set)
    clusters_map = {}  # root -> set of addresses

    def find(a):
        if a not in parent:
            parent[a] = a
            clusters_map[a] = {a}
            return a
        # Path compression
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        if not (allowed(a) and allowed(b)):
            return
        adjacency[a].add(b)
        adjacency[b].add(a)
        ra = find(a)
        rb = find(b)
        if ra != rb:
            # Union by size
            if len(clusters_map[ra]) < len(clusters_map[rb]):
                ra, rb = rb, ra
            clusters_map[ra] |= clusters_map[rb]
            del clusters_map[rb]
            parent[rb] = ra

    # ----------------------------------
    # 3. HOLDERS (LIVE STATE)
    # ----------------------------------
    holders = {}

    def ensure(addr):
        if not allowed(addr):
            return None
        if addr not in holders:
            h = {
                "balance": 0,
                "total_invested": 0.0,
            }
            if track_trades:
                h["buys"] = []
                h["sells"] = []
            holders[addr] = h
        return holders[addr]

    # ----------------------------------
    # 4. MAIN LOOP — OVER SNAPSHOTS
    # ----------------------------------
    holders_snaps = []
    clusters_snaps = []
    snapshotTimes = []

    tx_index = 0
    n_txs = len(all_txs)
    total_snaps = len(x_axis_sorted)
    last_snapshot_ts = None

    for i, ts_snap in enumerate(x_axis_sorted):
        if verbose and i % 5000 == 0:
            print(f"Progress: {i}/{total_snaps} ({(i/total_snaps)*100:.1f}%)")

        # Process all transactions up to current snapshot time
        while tx_index < n_txs and all_txs[tx_index]["timeStamp"] <= ts_snap:
            tx = all_txs[tx_index]
            tx_index += 1

            sender = tx["from"]
            receiver = tx["to"]
            amount = int(tx["value"])
            t = tx["timeStamp"]
            tx_hash = tx.get("hash", "")  # ← Get transaction hash

            # ✅ FAST NEAREST PRICE LOOKUP USING BISECT (O(log M))
            pos = bisect.bisect_right(x_axis_sorted, t)
            if pos == 0:
                idx = 0
            elif pos == n_prices:
                idx = n_prices - 1
            else:
                t_left = x_axis_sorted[pos - 1]
                t_right = x_axis_sorted[pos]
                # Pick closer timestamp
                idx = pos - 1 if (t - t_left) <= (t_right - t) else pos
            price = y_axis_ref[idx]
            invested_value = price * amount

            h_s = ensure(sender)
            h_r = ensure(receiver)

            # --- SENDER: decrease balance ---
            if h_s is not None and sender != ZERO:
                bal = h_s["balance"]
                if bal > 0:
                    avg_cost = h_s["total_invested"] / bal
                    used = min(bal, amount)
                    h_s["total_invested"] -= avg_cost * used
                    if h_s["total_invested"] < 0:
                        h_s["total_invested"] = 0.0
                h_s["balance"] -= amount

            # --- RECEIVER: increase balance ---
            if h_r is not None:
                h_r["balance"] += amount
                h_r["total_invested"] += invested_value

            # --- CLASSIFY BUYS/SELLS (only if tracking) ---
            if track_trades:
                sender_in_pair = sender in pair_set
                receiver_in_pair = receiver in pair_set
                if h_r is not None and sender_in_pair and not receiver_in_pair:
                    h_r["buys"].append({
                        "amount": amount,
                        "timestamp": t,
                        "price": price,
                        "value": invested_value,
                        "hash": tx_hash,
                    })
                if h_s is not None and receiver_in_pair and not sender_in_pair:
                    h_s["sells"].append({
                        "amount": amount,
                        "timestamp": t,
                        "price": price,
                        "value": invested_value,
                        "hash": tx_hash,
                    })

            # --- UPDATE CLUSTER GRAPH ---
            union(sender, receiver)

        # --- SNAPSHOT DELTA FILTER ---
        if last_snapshot_ts is not None and min_snapshot_delta > 0:
            if ts_snap - last_snapshot_ts < min_snapshot_delta:
                continue
        last_snapshot_ts = ts_snap

        # --- UPDATE HOLDER METADATA (avg_price, link count) ---
        for addr, h in holders.items():
            bal = h["balance"]
            h["avg_price"] = (h["total_invested"] / bal) if bal > 0 else 0.0
            h["cluster_link_count"] = len(adjacency.get(addr, ()))

        # --- BUILD CLUSTERS SNAPSHOT ---
        clusters = []
        for members in clusters_map.values():
            total_bal = 0
            total_inv = 0.0
            for addr in members:
                h = holders.get(addr)
                if h:
                    total_bal += h["balance"]
                    total_inv += h["total_invested"]
            avg_p = total_inv / total_bal if total_bal > 0 else 0.0
            cluster_entry = {
                "balance": total_bal,
                "total_invested": total_inv,
                "avg_price": avg_p,
            }
            if snapshot_full:
                cluster_entry["members"] = set(members)
            else:
                cluster_entry["member_count"] = len(members)
            clusters.append(cluster_entry)

        # --- STORE HOLDER SNAPSHOT ---
        if snapshot_full:
            snap_holders = {}
            for addr, h in holders.items():
                h_copy = {
                    "balance": h["balance"],
                    "total_invested": h["total_invested"],
                    "avg_price": h["avg_price"],
                    "cluster_link_count": h["cluster_link_count"],
                }
                if track_trades:
                    # ❗ FIX: copy lists so old snapshots don't mutate
                    h_copy["buys"] = [trade.copy() for trade in h["buys"]]
                    h_copy["sells"] = [trade.copy() for trade in h["sells"]]
                snap_holders[addr] = h_copy
        else:
            snap_holders = {}
            for addr, h in holders.items():
                entry = {
                    "balance": h["balance"],
                    "avg_price": h["avg_price"],
                    "cluster_link_count": h["cluster_link_count"],
                }
                snap_holders[addr] = entry

        holders_snaps.append(snap_holders)
        clusters_snaps.append(clusters)
        snapshotTimes.append(ts_snap)

    if verbose:
        print("Progress: 100% — DONE!")
    return holders_snaps, clusters_snaps, snapshotTimes

def balance_weighted_avg_price(obj):
    total_weight = 0
    total_value = 0

    # Case 1: holders (dict)
    if isinstance(obj, dict):
        for addr, info in obj.items():
            bal = info.get("balance", 0)
            avg_p = info.get("avg_price", 0)

            if bal > 0:
                total_weight += bal
                total_value += bal * avg_p

    # Case 2: clusters (list)
    elif isinstance(obj, list):
        for cluster in obj:
            bal = cluster.get("balance", 0)
            avg_p = cluster.get("avg_price", 0)

            if bal > 0:
                total_weight += bal
                total_value += bal * avg_p

    else:
        raise TypeError("Input must be holders(dict) or clusters(list).")

    if total_weight == 0:
        return 0

    return total_value / total_weight

def build_bwap_series(snapshots, timestamps):
    """
    snapshots  : list of holder-dicts OR list of cluster-lists
    timestamps : list of timestamps (same length as snapshots)
    
    Returns:
        (timestamps, bwap_axis)
    """
    bwap_axis = []

    for snap in snapshots:
        bwap_axis.append(balance_weighted_avg_price(snap))

    return timestamps, bwap_axis

import csv

def load_dune_data_csv(filename="dune_results_complete.csv"):
    with open(filename, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)

def count_over_threshold(snaps, threshold):
    """
    For each snapshot, return how many holders/clusters
    have balance > threshold.

    Works for both:
      - holders_snaps: list of dicts {address -> holder_data}
      - clusters_snaps: list of lists of cluster dicts
    """
    counts = []

    for snap in snaps:

        # Case 1: holders_snaps (dict form)
        if isinstance(snap, dict):
            count = sum(
                1 for h in snap.values()
                if h.get("balance", 0) > threshold
            )

        # Case 2: clusters_snaps (list form)
        else:
            count = sum(
                1 for c in snap
                if c.get("balance", 0) > threshold
            )

        counts.append(count)

    return counts

def count_new_holders(holders_snaps, threshold):
    """
    For each snapshot, count how many addresses become *holders* in that snapshot.

    Definition:
      - holder in a snapshot: balance > threshold
      - new holder at snapshot i: holder in snapshot i, but NOT a holder in snapshot i-1

    This allows:
      - addr was holder, then dropped below threshold or to zero,
        then later goes above threshold again -> counted as new again.
    """
    prev_holders = set()   # addresses that were holders in the previous snapshot
    counts = []

    for snap in holders_snaps:
        # addresses that are holders in this snapshot
        current_holders = {
            addr for addr, h in snap.items()
            if h.get("balance", 0) > threshold
        }

        # new = in current, but not in previous
        new_holders = current_holders - prev_holders
        counts.append(len(new_holders))

        # update for next iteration
        prev_holders = current_holders

    return counts




#Example usage:
if __name__ == "__main__":
    token0 = '0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2'
    token1 = '0x9778ac3d5a2f916aa9abf1eb85c207d990ca2655'
    pair = '0x7bdb64fb70adb01c7bbfed98e4def7df70c8318b'
    maxLen = 10000000
    [x_axis, y_axis, liq, pool] = buildPoolChart(token0,token1,pair,maxLen , clean = 1)
    all_txs = getAllTransfers(token1, maxLen, 0, 30164015, holder_address=None)
    
    #Load all known EVM CEX addresses
    all_cex_dict = load_dune_data_csv("dune_known_cex_addresses.csv")
    all_cex_addresses = [row['address'] for row in all_cex_dict]
    
    pair_addresses = pair
    cex_addresses = all_cex_addresses
    router_addresses = [
        '0x3fc91a3afd70395cd496c647d5a6cc9d4b2b7fad',  # uniswap_universal_router
        '0x74de5d4fcbf63e00296fd95d33236b9794016631',  # metamask_swap_router
        '0xb1ca6e0283503d2bd17c7a94c57f5f556bc42179', # OGSM V3 Pool
        '0x66a9893cc07d91d95644aedd05d03f95e1dba8af', # UniswapV4 router
        '0x6131b5fae19ea4f9d964eac0408e4408b66337b5', #KyberSwap Meta Aggregator
        '0x1111111254eeb25477b68fb85ed929f73a960582', #Aggregation Router V5
        '0xe37e799d5077682fa0a244d46e5649f71457bd09', #1Inch something
        '0xe37e799d5077682fa0a244d46e5649f71457bd09', #1Inch something
        '0x5141b82f5ffda4c6fe1e372978f1c5427640a190', #1Inch something
        '0x9008D19f58AAbD9eD0D60971565AA8510560ab41', #CoW Protocol: GPv2Settlement
        '0xa88800cd213da5ae406ce248380802bd53b47647', #1Inch: Settlement
        '0x70bf6634ee8cb27d04478f184b9b8bb13e5f4710', #0x: Settler 1.6
        '0x22f9dcf4647084d6c31b2765f6910cd85c178c18', #0x: Echange Proxy Flash...
        '0x3451b6b219478037a1ac572706627fc2bda1e812', #1Inch something
        '0x111111125421ca6dc452d289314280a0f8842a65', #1Inch: Aggregation Router V6
        '0xf081470f5c6fbccf48cc4e5b82dd926409dcdd67', #Kyber something
        '0x9008d19f58aabd9ed0d60971565aa8510560ab41', #GPv2Settlement
        '0x96c195f6643a3d797cb90cb6ba0ae2776d51b5f3', #0x: Exchange Proxy Flash Wallet
        '0x82d88875d64d60cbe9cbea47cb960ae0f04ebd4d', #0x: Protocol Settler
        '0x5418226af9c8d5d287a78fbbbcd337b86ec07d61', #0x: Protocol Settler
        '0x663dc15d3c1ac63ff12e45ab68fea3f0a883c251', #deBridge: Crosschain Forwarder Proxy
        '0xdf31a70a21a1931e02033dbba7deace6c45cfd0f', #0x: Protocol Settler
        '0xa7ca2c8673bcfa5a26d8ceec2887f2cc2b0db22a', #"weief wesllet"
        '0xb300000b72deaeb607a12d5f54773d1c19c7028d', #Binance dex router
        '0x1231deb6f5749ef6ce6943a275a1d3e7486f4eae', #LiFi Diamond
        '0x9642b23ed1e01df1092b92641051881a322f5d4e', #MEXC 16
        '0xc38e4e6a15593f908255214653d3d947ca1c2338', #Mayan: Shift
        '0xd152f549545093347a162dce210e7293f1452150', #Disperse.app
        '0x8fca4ade3a517133ff23ca55cdaea29c78c990b8', #Poloniex 7
        '0xcf5540fffcdc3d510b18bfca6d2b9987b0772559', #Odos: Router V2
        '0xad3b67bca8935cb510c8d18bd45f0b94f54a968f', #1Inch: Fusuin Resolver
        '0x00000047bb99ea4d791bb749d970de71ee0b1a34', #TransitSwap v5: Router
        '0xf81b45b1663b7ea8716c74796d99bbe4ea26f488', #Ourbit 1 Exchange
        '0x00000688768803bbd44095770895ad27ad6b0d95', #The T Resolver: Proxy (1Inch)
        '0x69460570c93f9de5e2edbc3052bf10125f0ca22d', #Rango V2: Rango Diamond
        '0x00000000009726632680fb29d3f7a9734e3010e2', #Rainbow router
    ]
    print('Building holders dict')
    start_time = time.time()
    holders_snaps, clusters_snaps, snapshotTimes  = buildHoldersDict(all_txs, x_axis, y_axis, pair_addresses, cex_addresses, router_addresses,
                                                                     avg_time_threshold=3600*24, trxCountThreshold=50, 
                                                                     min_snapshot_delta = 3600*12, track_trades=True, 
                                                                     snapshot_full=True, verbose = True )
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")

    print('Displaying top cluster links')
    snapshot = holders_snaps[-1]   # pick latest snapshot (or any index)
    top10 = sorted(
        snapshot.items(),
        key=lambda x: x[1].get("cluster_link_count", 0),
        reverse=True
    )[:10]
    for addr, info in top10:
        print(addr, info.get("cluster_link_count", 0))

    print('Getting bwap series')
    timestamps, bwap_holders = build_bwap_series(holders_snaps, snapshotTimes)
    timestamps, bwap_clusters = build_bwap_series(clusters_snaps, snapshotTimes)
    
    print('Getting new holder counts')
    HolderCount     = count_over_threshold(holders_snaps, 100)
    NewHoldersCount = count_new_holders(holders_snaps, 100)
    
    print('Plotting chart')
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # -----------------------------------------
    # 1) PRIMARY AXIS (LEFT): PRICE + BWAP
    # -----------------------------------------
    ax1.plot(x_axis, y_axis, label='Price', linewidth=1.8, color="black")
    ax1.plot(snapshotTimes, bwap_holders,  label="BWAP (holders)",  linewidth=1.8, color="blue")
    ax1.plot(snapshotTimes, bwap_clusters, label="BWAP (clusters)", linewidth=1.8, color="red")
    
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Price")
    ax1.grid(False)
    
    
    # -----------------------------------------
    # 2) SECOND AXIS (RIGHT): TOTAL HOLDERS
    # -----------------------------------------
    ax2 = ax1.twinx()
    ax2.set_ylabel("Holder Count")
    
    snap_x = np.array(snapshotTimes)
    
    # Total holders as filled area
    ax2.fill_between(
        snap_x,
        HolderCount,
        step="mid",
        alpha=0.25,
        color="orange",
        label="Holders > threshold"
    )
    
    
    # -----------------------------------------
    # 3) THIRD AXIS (RIGHT, OFFSET): NEW HOLDERS
    # -----------------------------------------
    ax3 = ax1.twinx()
    
    # Offset the third axis so it doesn't overlap with ax2
    ax3.spines["right"].set_position(("axes", 1.12))
    ax3.set_frame_on(True)
    ax3.patch.set_visible(False)
    
    for sp in ax3.spines.values():
        sp.set_visible(True)
    
    ax3.set_ylabel("New Holders")
    
    # Very visible bold line for new holders
    ax3.plot(
        snap_x,
        NewHoldersCount,
        color="green",
        linewidth=3,
        label="New holders > threshold"
    )
    
    
    # -----------------------------------------
    # Combined legend collecting from all axes
    # -----------------------------------------
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    
    ax1.legend(
        lines1 + lines2 + lines3,
        labels1 + labels2 + labels3,
        loc="upper left"
    )
    
    plt.title("Price, BWAP, Holder Count and New Holders (3-axis view)")
    plt.tight_layout()
    plt.show()
