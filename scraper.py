# scraper.py — Real-Time Flipkart Pricing Scraper
# Run: python scraper.py

import requests, redis, json, time
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime

RAPIDAPI_KEY  = "9c4b954211msha8b4053a645f193p1d9f9djsndb507fe180a5"
RAPIDAPI_HOST = "real-time-flipkart-data2.p.rapidapi.com"
BASE_URL      = f"https://{RAPIDAPI_HOST}"

HEADERS = {
    "x-rapidapi-key":  RAPIDAPI_KEY,
    "x-rapidapi-host": RAPIDAPI_HOST,
    "Content-Type":    "application/json"
}

CATEGORIES = {
    "mobiles":      "tyy,4io",
    "laptops":      "6bo,b5g",
    "televisions":  "ckf,czl",
    "audio":        "0pm",        # compound ID broken, single works
    "tablets":      "itl,tdl",
    "smartwatch":   "ajy,eim",    # correct Flipkart smartwatch SID
    "cameras":      "6bo",        # 6bo,c1r broken, 6bo alone works
    "electronics":  "tyy",
}

# ONE fallback ID per stubborn category — tried only if primary returns < 4
# No search fallback (endpoint disabled on this API plan)
CAT_ALT_IDS = {
    "audio":       ["tbl"],
    "tablets":     ["itl", "tdl"],
    "smartwatch":  ["byf,ri7"],
}

MAX_HISTORY = 100

# ── Redis ──────────────────────────────────────────────────────────────────────
try:
    r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    r.ping()
    REDIS_ON = True
    print("Redis connected")
except:
    REDIS_ON = False
    _mem = {}
    print("Redis unavailable — using memory cache")

def cache_set(key, value, ttl=360):
    data = json.dumps(value, default=str)
    if REDIS_ON: r.setex(key, ttl, data)
    else: _mem[key] = data

def cache_get(key):
    raw = r.get(key) if REDIS_ON else _mem.get(key)
    return json.loads(raw) if raw else None

# ── Normalise one raw API item into a clean product dict ──────────────────────
def _extract_price(item):
    """Try every field name the RapidAPI search/category endpoints might use."""
    for field in ("price", "selling_price", "discounted_price",
                  "final_price", "special_price", "offer_price",
                  "current_price", "unit_price"):
        val = item.get(field)
        if val is not None:
            try:
                p = float(str(val).replace(",", "").replace("₹", "").strip())
                if p > 0:
                    return p
            except (ValueError, TypeError):
                continue
    # last resort: price_str
    for field in ("price_str", "mrp_str", "display_price"):
        val = item.get(field)
        if val:
            try:
                p = float(str(val).replace(",", "").replace("₹", "").strip())
                if p > 0:
                    return p
            except (ValueError, TypeError):
                continue
    return None

def _extract_mrp(item, price_fallback):
    """Try every MRP field name."""
    for field in ("mrp", "original_price", "retail_price",
                  "maximum_retail_price", "base_price", "list_price"):
        val = item.get(field)
        if val is not None:
            try:
                m = float(str(val).replace(",", "").replace("₹", "").strip())
                if m >= price_fallback:
                    return m
            except (ValueError, TypeError):
                continue
    return price_fallback

def _parse_item(item, _debug=False):
    price = _extract_price(item)
    if price is None:
        if _debug:
            print(f"    [PARSE-FAIL] keys={list(item.keys())[:10]}")
        return None

    mrp = _extract_mrp(item, price)

    # name: try multiple fields
    name = ""
    for nf in ("title", "name", "product_name", "product_title"):
        name = item.get(nf, "")
        if name:
            break

    # brand
    brand = item.get("brand") or item.get("brand_name") or "Unknown"

    # rating
    rating = item.get("rating", {})
    if isinstance(rating, (int, float)):
        avg_r, cnt_r = float(rating), 0
    elif isinstance(rating, dict):
        avg_r = float(rating.get("average", rating.get("avg", 4.0)))
        cnt_r = int(rating.get("count", rating.get("total", 0)))
    else:
        avg_r, cnt_r = 4.0, 0

    # image
    images = item.get("images") or item.get("image_urls") or []
    image  = (images[0] if isinstance(images, list) and images
              else item.get("image", item.get("thumbnail", "")))

    return {
        "pid":          item.get("pid") or item.get("id") or item.get("product_id", ""),
        "name":         str(name)[:70],
        "brand":        str(brand),
        "price":        round(price, 2),
        "mrp":          round(mrp, 2),
        "discount_pct": round((1 - price / mrp) * 100, 1) if mrp else 0.0,
        "url":          item.get("url") or item.get("product_url", ""),
        "image":        str(image),
        "stock":        item.get("stock", "IN_STOCK"),
        "rating":       round(avg_r, 1),
        "rating_count": cnt_r,
        "timestamp":    datetime.now().isoformat(),
    }

# ── Fetch products by Flipkart category ID ────────────────────────────────────
def fetch_category_products(category_id, n=10):
    try:
        res  = requests.get(
            f"{BASE_URL}/products-by-category",
            headers=HEADERS,
            params={"page": "1", "categoryId": category_id, "sortBy": "POPULARITY"},
            timeout=15
        )
        data = res.json()
        if not data.get("success"):
            return []

        raw = data.get("data", [])

        # API returns either a list of products OR a single product dict
        if isinstance(raw, dict):
            raw = [raw]
        if not isinstance(raw, list):
            return []

        products = []
        for item in raw[:n]:
            p = _parse_item(item)
            if p: products.append(p)
        return products
    except Exception as e:
        print(f"  [ERROR] {e}")
        return []

# ── Search any keyword on Flipkart ────────────────────────────────────────────
def search_flipkart_products(query: str, n: int = 12, _debug: bool = False) -> list:
    """Search Flipkart by keyword via /product-search endpoint."""
    try:
        res = requests.get(
            f"{BASE_URL}/product-search",
            headers=HEADERS,
            params={"q": query, "page": "1", "sort_by": "RELEVANCE"},
            timeout=15,
        )
        data = res.json()
        if not data.get("success"):
            msg = data.get("message", data.get("msg", str(data)))
            print(f"    [SEARCH-FAIL] API says: {msg!r}")
            return []

        # Response: {"data": {"products": [...], "total": N, ...}}
        inner = data.get("data", {})
        if isinstance(inner, dict):
            items = inner.get("products", inner.get("result", []))
        elif isinstance(inner, list):
            items = inner
        else:
            items = []

        if _debug and items:
            print(f"    [SEARCH-DEBUG] {len(items)} raw items, keys: {list(items[0].keys())[:12]}")

        products = []
        for item in items[:n]:
            p = _parse_item(item, _debug=_debug)
            if p:
                products.append(p)
        return products
    except Exception as e:
        print(f"  [SEARCH ERROR] {e}")
        return []

# ── Detect category from query string ─────────────────────────────────────────
def detect_category(query: str) -> str:
    q = query.lower()
    if any(w in q for w in ["mobile","phone","smartphone","iphone","samsung","oneplus","redmi","realme","vivo","oppo"]):
        return "mobiles"
    if any(w in q for w in ["laptop","notebook","macbook","chromebook"]):
        return "laptops"
    if any(w in q for w in ["tv","television","qled","oled","smart tv"]):
        return "televisions"
    if any(w in q for w in ["headphone","earphone","speaker","audio","boat","neckband","tws"]):
        return "audio"
    if any(w in q for w in ["tablet","ipad"]):
        return "tablets"
    if any(w in q for w in ["watch","smartwatch"]):
        return "smartwatch"
    if any(w in q for w in ["camera","dslr","mirrorless"]):
        return "cameras"
    return "electronics"

# ── Build pricing-ready product dict from a list + index of focal product ──────
def build_product_pricing_data(products: list, focal_idx: int = 0) -> dict | None:
    """
    Given a list of raw products, take products[focal_idx] as focal and
    the next available ones as competitors.  Returns a dict ready for /predict.
    """
    if not products:
        return None
    focal = products[focal_idx]
    # Competitors = other products in the list (skip focal)
    comps = [p for i, p in enumerate(products) if i != focal_idx]
    # Pad if fewer than 3 competitors
    while len(comps) < 3:
        comps.append(focal)
    comps = comps[:3]

    return {
        "pid":          focal["pid"],
        "product_name": focal["name"],
        "brand":        focal["brand"],
        "product_url":  focal["url"],
        "image":        focal["image"],
        "stock":        focal["stock"],
        "rating":       focal["rating"],
        "rating_count": focal["rating_count"],
        "market_price": focal["price"],
        "mrp":          focal["mrp"],
        "discount_pct": focal["discount_pct"],
        "comp_1":       comps[0]["price"],
        "comp_2":       comps[1]["price"],
        "comp_3":       comps[2]["price"],
        "comp_avg":     round(sum(c["price"] for c in comps) / 3, 2),
        "comp_min":     round(min(c["price"] for c in comps), 2),
        "comp_max":     round(max(c["price"] for c in comps), 2),
        "competitors": [
            {"pid": c["pid"], "name": c["name"][:50], "brand": c["brand"],
             "price": c["price"], "mrp": c["mrp"],
             "rating": c["rating"], "url": c["url"]}
            for c in comps
        ],
        "timestamp": datetime.now().isoformat(),
    }

# ── Legacy single-focal builder (used by scrape_and_cache) ────────────────────
def build_product_data(products, label, cat_id):
    if len(products) < 4:
        return None
    focal = products[0]
    comps = products[1:4]
    return {
        "pid":          focal["pid"],
        "product_name": focal["name"],
        "brand":        focal["brand"],
        "product_url":  focal["url"],
        "image":        focal["image"],
        "stock":        focal["stock"],
        "rating":       focal["rating"],
        "rating_count": focal["rating_count"],
        "category":     label,
        "category_id":  cat_id,
        "market_price": focal["price"],
        "mrp":          focal["mrp"],
        "discount_pct": focal["discount_pct"],
        "comp_1":   comps[0]["price"],
        "comp_2":   comps[1]["price"],
        "comp_3":   comps[2]["price"],
        "comp_avg": round(sum(c["price"] for c in comps) / 3, 2),
        "comp_min": round(min(c["price"] for c in comps), 2),
        "comp_max": round(max(c["price"] for c in comps), 2),
        "competitors": [
            {"pid": c["pid"], "name": c["name"][:50], "brand": c["brand"],
             "price": c["price"], "mrp": c["mrp"],
             "rating": c["rating"], "url": c["url"]}
            for c in comps
        ],
        "timestamp": datetime.now().isoformat(),
    }

# ── Search: cache full result list + return ────────────────────────────────────
def search_and_cache_all(query: str, n: int = 12) -> list:
    """
    Search Flipkart for query, cache the full list under
    'search_list:<query>', and return the list.
    """
    products = search_flipkart_products(query, n=n)
    if products:
        key = f"search_list:{query.lower().strip()}"
        cache_set(key, products, ttl=300)
        print(f"  [SEARCH] '{query}' → {len(products)} results")
    return products

def get_search_list(query: str) -> list:
    key = f"search_list:{query.lower().strip()}"
    return cache_get(key) or []

# ── Legacy single-result search (kept for api /search endpoint) ───────────────
def build_product_data_from_search(query: str, products: list) -> dict | None:
    if len(products) < 4:
        return None
    cat   = detect_category(query)
    focal = products[0]
    comps = products[1:4]
    return {
        "pid":          focal["pid"],
        "product_name": focal["name"],
        "brand":        focal["brand"],
        "product_url":  focal["url"],
        "image":        focal["image"],
        "stock":        focal["stock"],
        "rating":       focal["rating"],
        "rating_count": focal["rating_count"],
        "category":     cat,
        "category_id":  "search",
        "market_price": focal["price"],
        "mrp":          focal["mrp"],
        "discount_pct": focal["discount_pct"],
        "comp_1":   comps[0]["price"],
        "comp_2":   comps[1]["price"],
        "comp_3":   comps[2]["price"],
        "comp_avg": round(sum(c["price"] for c in comps) / 3, 2),
        "comp_min": round(min(c["price"] for c in comps), 2),
        "comp_max": round(max(c["price"] for c in comps), 2),
        "competitors": [
            {"pid": c["pid"], "name": c["name"][:50], "brand": c["brand"],
             "price": c["price"], "mrp": c["mrp"],
             "rating": c["rating"], "url": c["url"]}
            for c in comps
        ],
        "timestamp": datetime.now().isoformat(),
        "_search_query": query,
    }

def search_and_cache(query: str) -> dict | None:
    products  = search_flipkart_products(query, n=8)
    prod_data = build_product_data_from_search(query, products)
    if prod_data:
        key = f"search:{query.lower().strip()}"
        cache_set(key, prod_data, ttl=300)
        update_sku_history(
            pid=prod_data["pid"],
            market_price=prod_data["market_price"],
            comp_avg=prod_data["comp_avg"],
            mrp=prod_data["mrp"],
        )
        print(f"  [SEARCH] '{query}' → {prod_data['brand']} ₹{prod_data['market_price']:,.0f}")
    return prod_data

def get_search_result(query: str) -> dict | None:
    return cache_get(f"search:{query.lower().strip()}")

# ── Category product list cache ────────────────────────────────────────────────
def get_category_product_list(label: str) -> list:
    """Return full cached product list for a category."""
    return cache_get(f"cat_list:{label}") or []

# ── SKU history ────────────────────────────────────────────────────────────────
def update_sku_history(pid, market_price, comp_avg, mrp, optimal_price=None):
    key     = f"history:{pid}"
    history = cache_get(key) or []
    history.append({
        "time":          datetime.now().strftime("%H:%M:%S"),
        "date":          datetime.now().strftime("%Y-%m-%d"),
        "market_price":  round(market_price, 2),
        "comp_avg":      round(comp_avg, 2),
        "mrp":           round(mrp, 2),
        "optimal_price": round(optimal_price, 2) if optimal_price else None,
    })
    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]
    cache_set(key, history, ttl=86400)
    return history

def get_sku_history(pid):
    return cache_get(f"history:{pid}") or []

# ── Main scrape loop ───────────────────────────────────────────────────────────
def scrape_and_cache():
    print(f"\n[{datetime.now():%H:%M:%S}] Fetching live Flipkart prices...")
    all_products = {}

    for label, cat_id in CATEGORIES.items():
        products = fetch_category_products(cat_id, n=10)
        time.sleep(1.5)   # be gentle with the API

        # Try alt IDs only for known-problematic categories
        if len(products) < 4 and label in CAT_ALT_IDS:
            for alt_id in CAT_ALT_IDS[label]:
                alt_prods = fetch_category_products(alt_id, n=10)
                time.sleep(1.5)
                if len(alt_prods) > len(products):
                    products = alt_prods
                if len(products) >= 4:
                    break

        # NOTE: /product-search is disabled on this API plan — no search fallback

        prod_data = build_product_data(products, label, cat_id)

        if products:
            cache_set(f"cat_list:{label}", products, ttl=600)

        if prod_data:
            prod_data["category"]    = label
            prod_data["category_id"] = cat_id
            all_products[label]      = prod_data
            update_sku_history(
                pid=prod_data["pid"],
                market_price=prod_data["market_price"],
                comp_avg=prod_data["comp_avg"],
                mrp=prod_data["mrp"],
            )
            print(f"  {label:15} → {prod_data['brand']:18} "
                  f"₹{prod_data['market_price']:>8,.0f} "
                  f"(MRP ₹{prod_data['mrp']:>8,.0f}, {prod_data['discount_pct']:.0f}% off) "
                  f"| {prod_data['product_name'][:35]} [{len(products)} items]")
        else:
            print(f"  {label:15} → no data (category ID may need updating)")

    if all_products:
        cache_set("live_products", all_products, ttl=600)
        cache_set("last_scrape",   datetime.now().isoformat(), ttl=1200)
        print(f"\n  Cached {len(all_products)} categories\n")

    if all_products:
        cache_set("live_products", all_products, ttl=360)
        cache_set("last_scrape",   datetime.now().isoformat(), ttl=600)
        print(f"\n  Cached {len(all_products)} categories\n")

def get_live_products():
    return cache_get("live_products") or {}

def get_last_scrape():
    t = cache_get("last_scrape")
    return t[11:19] if isinstance(t, str) and len(t) > 19 else "never"

if __name__ == "__main__":
    scheduler = BackgroundScheduler()
    scheduler.add_job(scrape_and_cache, "interval", minutes=10)
    scheduler.start()
    scrape_and_cache()
    print("Scraper running — every 5 min. Ctrl+C to stop.")
    try:
        while True: time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()