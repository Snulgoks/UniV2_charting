import time, requests

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
        "apikey": "HGCJYT8W76BEZJKVTPEMUP8BYA9APJEQRR",
    }
    r = requests.get(url, params=params, timeout=30).json()
    if r.get("status") != "1" or not r.get("result"):
        print("first_tx_block error:", addr, "| msg:", r.get("message"), "| result:", r.get("result"))
        return None
    return int(r["result"][0]["blockNumber"])

def getERC20Transactions(address, startBlock, endBlock, contract_address=None, max_pages=1):
    api_keys = ["HGCJYT8W76BEZJKVTPEMUP8BYA9APJEQRR", "NTCY5CQPJYV71MABW6SCSMVVBNJPYXBK82"]
    base_url = "https://api.etherscan.io/v2/api"
    chainid = 1

    all_txs, page, offset, key_idx = [], 1, 10000, 0

    while page <= max_pages:
        params = {
            "chainid": chainid, "module": "account", "action": "tokentx",
            "address": str(address), "page": page, "offset": offset,
            "startblock": int(startBlock), "endblock": int(endBlock),
            "sort": "asc", "apikey": api_keys[key_idx],
        }
        if contract_address:
            params["contractaddress"] = str(contract_address)

        # retry a few times on timeouts/temporary issues
        for attempt in range(5):
            try:
                r = requests.get(base_url, params=params, timeout=60)
                r.raise_for_status()
                payload = r.json()
                break
            except requests.exceptions.ReadTimeout:
                time.sleep(1.5 * (attempt + 1))
            except requests.exceptions.RequestException:
                time.sleep(1.5 * (attempt + 1))
        else:
            raise RuntimeError(f"Etherscan request keeps timing out for blocks {startBlock}-{endBlock}")

        result = payload.get("result")
        status = payload.get("status")
        message = payload.get("message")

        if isinstance(result, str) and "rate limit" in result.lower():
            key_idx = (key_idx + 1) % len(api_keys)
            continue
        if status == "0" and (message == "No transactions found" or result == []):
            break
        if status != "1" or not isinstance(result, list):
            print("Etherscan error:", message, "| result:", result)
            break

        all_txs.extend(result)
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

import numpy as np

def clean_sandwich(x_axis, y_axis, factor=5):
    # ---- 1) GLOBAL OUTLIER REMOVAL (IQR) ----
    Q1 = np.percentile(y_axis, 25)
    Q3 = np.percentile(y_axis, 75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    mask = [(lower <= y <= upper) for y in y_axis]
    x = [x_axis[i] for i in range(len(x_axis)) if mask[i]]
    y = [y_axis[i] for i in range(len(y_axis)) if mask[i]]

    # ---- 2) LOCAL SANDWICH SPIKE REMOVAL ----
    clean_x = []
    clean_y = []

    for i in range(len(y)):
        if i == 0 or i == len(y) - 1:
            clean_x.append(x[i])
            clean_y.append(y[i])
            continue

        prev_y = y[i-1]
        next_y = y[i+1]
        cur_y  = y[i]

        if cur_y > max(prev_y, next_y) * factor:
            continue
        if cur_y < min(prev_y, next_y) / factor:
            continue

        clean_x.append(x[i])
        clean_y.append(y[i])

    return clean_x, clean_y

def buildPoolChart(token0, token1, pair, maxLen):
    token0 = token0.lower()
    token1 = token1.lower()
    pair   = pair.lower()

    data = getAllERC20Transactions(pair, maxLen, get_first_tx_block(pair), endBlock=25_164_015)
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
        
    [x_axis,y_axis] = clean_sandwich(x_axis,y_axis)
    
    return x_axis, y_axis, liq, pool




#Example usage:
if __name__ == "__main__":

    token0 = '0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2'
    token1 = '0x0f7dc5d02cc1e1f5ee47854d534d332a1081ccc8'
    pair = '0xf97503af8230a7e72909d6614f45e88168ff3c10'
    [x_axis, y_axis, liq, pool] = buildPoolChart(token0,token1,pair, 10000000)
    

    import matplotlib.pyplot as plt
    plt.plot(x_axis, y_axis)
    plt.show()
