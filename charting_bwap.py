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
        msg = payload.get("message")

        if status == "0" and (result == [] or msg == "No transactions found"):
            break
        if status != "1" or not isinstance(result, list):
            print("Etherscan error:", msg, "| result:", result)
            break

        all_txs.extend(result)
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


from bisect import bisect_left
from collections import defaultdict

def buildHoldersDict(
    all_txs,
    x_axis,
    y_axis,
    pair_addresses,
    avg_time_threshold=None,
    trxCountThreshold=None,
    min_snapshot_delta=0  # <--- NEW: minimum seconds between snapshots
):

    # ----------------------------------
    # 0. PREP FAST STRUCTURES
    # ----------------------------------
    pair_set = set(pair_addresses)
    ZERO = "0x0000000000000000000000000000000000000000"
    HUB = {"0x6ae83320fe7508489c3c2e2575e084c0af9689f1"}
    SKIP = pair_set | HUB | {ZERO}

    # Precompute maps for speed
    all_txs.sort(key=lambda x: int(x["timeStamp"]))
    x_axis_sorted = sorted(x_axis)

    # ----------------------------------
    # 1. ACTIVITY ANALYSIS (FAST)
    # ----------------------------------
    addr_ts = defaultdict(list)
    for tx in all_txs:
        t = int(tx["timeStamp"])
        addr_ts[tx["from"]].append(t)
        addr_ts[tx["to"]].append(t)

    addr_activity = {}
    for addr, times in addr_ts.items():
        times.sort()
        cnt = len(times)
        if cnt < 2:
            avg_int = None
        else:
            total_gap = 0
            prev = times[0]
            for i in range(1, cnt):
                cur = times[i]
                total_gap += cur - prev
                prev = cur
            avg_int = total_gap / (cnt - 1)
        addr_activity[addr] = (avg_int, cnt)

    def allowed(addr):
        if addr in SKIP:
            return False
        avg_i, cnt = addr_activity.get(addr, (None, 0))
        if avg_i is None:
            return True
        if avg_time_threshold is None or trxCountThreshold is None:
            return True
        return not (cnt > trxCountThreshold and avg_i < avg_time_threshold)

    # ----------------------------------
    # 2. UNION–FIND (VERY FAST)
    # ----------------------------------
    parent = {}
    adjacency = defaultdict(set)

    def find(a):
        # iterative with path compression
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        if not allowed(a) or not allowed(b):
            return
        adjacency[a].add(b)
        adjacency[b].add(a)
        parent.setdefault(a, a)
        parent.setdefault(b, b)
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    # ----------------------------------
    # 3. HOLDERS
    # ----------------------------------
    holders = {}

    def ensure(addr):
        if not allowed(addr):
            return None
        return holders.setdefault(addr, {
            "balance": 0,
            "total_invested": 0.0,
            "acquisitions": [],
            "buys": [],
            "sells": [],
            "avg_price": 0.0,
            "cluster_link_count": 0,
        })

    # ----------------------------------
    # 4. SNAPSHOT OUTPUT
    # ----------------------------------
    holders_snaps = []
    clusters_snaps = []
    snapshotTimes = []

    tx_index = 0
    n = len(all_txs)
    total_snaps = len(x_axis_sorted)

    last_snapshot_ts = None  # <--- NEW: track last snapshot timestamp

    # ----------------------------------
    # 5. MAIN LOOP — ULTRA FAST SCAN
    # ----------------------------------
    for i, ts_snap in enumerate(x_axis_sorted):

        # progress print (lightweight)
        if i % 5000 == 0:
            print(f"Progress: {i}/{total_snaps} ({(i/total_snaps)*100:.1f}%)")

        # process all tx up to this snapshot time
        while tx_index < n and int(all_txs[tx_index]["timeStamp"]) <= ts_snap:
            tx = all_txs[tx_index]
            tx_index += 1

            sender = tx["from"]
            receiver = tx["to"]
            amount = int(tx["value"])
            t = int(tx["timeStamp"])

            # price lookup using sorted x_axis
            pos = bisect_left(x_axis_sorted, t)
            if pos == 0:
                price = y_axis[0]
            elif pos == len(x_axis_sorted):
                price = y_axis[-1]
            else:
                before = x_axis_sorted[pos - 1]
                after = x_axis_sorted[pos]
                price = y_axis[pos - 1] if abs(t - before) < abs(t - after) else y_axis[pos]

            invested_value = price * amount

            h_s = ensure(sender)
            h_r = ensure(receiver)

            # sender loses
            if h_s and sender != ZERO:
                bal = h_s["balance"]
                if bal > 0:
                    avg_cost = h_s["total_invested"] / bal if h_s["total_invested"] > 0 else 0
                    used = min(bal, amount)
                    h_s["total_invested"] -= avg_cost * used
                    if h_s["total_invested"] < 0:
                        h_s["total_invested"] = 0
                h_s["balance"] -= amount

            # receiver gains
            if h_r:
                h_r["balance"] += amount
                h_r["total_invested"] += invested_value
                h_r["acquisitions"].append({
                    "amount": amount,
                    "timestamp": t,
                    "price": price,
                    "value": invested_value,
                })

            # classify buys/sells vs pair
            if h_r and sender in pair_set and receiver not in pair_set:
                h_r["buys"].append({
                    "amount": amount,
                    "timestamp": t,
                    "price": price,
                    "value": invested_value,
                })
            if h_s and receiver in pair_set and sender not in pair_set:
                h_s["sells"].append({
                    "amount": amount,
                    "timestamp": t,
                    "price": price,
                    "value": invested_value,
                })

            # cluster graph
            union(sender, receiver)

        # -------------------------
        # SNAPSHOT DECISION
        # -------------------------
        if last_snapshot_ts is not None and min_snapshot_delta > 0:
            # skip snapshot if not enough time passed
            if ts_snap - last_snapshot_ts < min_snapshot_delta:
                continue

        # -------------------------
        # CREATE SNAPSHOT
        # -------------------------
        last_snapshot_ts = ts_snap

        # update avg_price per holder
        for addr, h in holders.items():
            bal = h["balance"]
            h["avg_price"] = (h["total_invested"] / bal) if bal > 0 else 0.0

        # build clusters from union-find
        clusters_map = defaultdict(set)
        for a in parent:
            clusters_map[find(a)].add(a)

        clusters = []
        for members in clusters_map.values():
            total_bal = 0
            total_inv = 0.0
            for addr in members:
                h = holders.get(addr)
                if h:
                    total_bal += h["balance"]
                    total_inv += h["total_invested"]
            avg_p = (total_inv / total_bal) if total_bal else 0.0
            clusters.append({
                "members": set(members),  # copy
                "balance": total_bal,
                "total_invested": total_inv,
                "avg_price": avg_p,
            })

        # cluster_link_count
        for addr, h in holders.items():
            h["cluster_link_count"] = len(adjacency.get(addr, ()))

        # store snapshot
        holders_snaps.append({k: v.copy() for k, v in holders.items()})
        clusters_snaps.append(clusters)
        snapshotTimes.append(ts_snap)

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








#Example usage:
if __name__ == "__main__":
    token0 = '0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2'
    token1 = '0xe0f63a424a4439cbe457d80e4f4b51ad25b2c56c'
    pair = '0x52c77b0cb827afbad022e6d6caf2c44452edbc39'
    maxLen = 10000000
    [x_axis, y_axis, liq, pool] = buildPoolChart(token0,token1,pair,maxLen , clean = 1)
    all_txs = getAllTransfers(token1, maxLen, 0, 30164015, holder_address=None)
    pair_addresses = [
        pair,
        token1,
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
        '0xc38e4e6a15593f908255214653d3d947ca1c2338' #Mayan: Shift
    ]
    print('Building holders dict')
    holders_snaps, clusters_snaps, snapshotTimes  = buildHoldersDict(all_txs, x_axis, y_axis, pair_addresses,avg_time_threshold=3600*12, trxCountThreshold=100, min_snapshot_delta = 3600)

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
    
    print('Plotting chart')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,6))
    plt.plot(x_axis,y_axis, label = 'price')
    plt.plot(snapshotTimes, bwap_holders,  label="BWAP (holders)",  color="blue", linewidth=1.8)
    plt.plot(snapshotTimes, bwap_clusters, label="BWAP (clusters)", color="red",  linewidth=1.8)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
