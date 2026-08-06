# api.py — Real-Time Causal Pricing Engine (Flipkart)
# Run: uvicorn api:app --reload --port 8000

import json, pickle, time, asyncio, hashlib, os, warnings
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")

app = FastAPI(title="Real-Time Causal Pricing Engine — Flipkart")
app.add_middleware(CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Redis ──────────────────────────────────────────────────────────────────────
import redis as redis_lib
try:
    _r = redis_lib.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    _r.ping()
    REDIS_ON = True
except:
    REDIS_ON = False
    _mem_cache = {}

def cache_get(key):
    val = _r.get(key) if REDIS_ON else _mem_cache.get(key)
    return json.loads(val) if val else None

def cache_set(key, value, ttl=60):
    data = json.dumps(value, default=str)
    if REDIS_ON: _r.setex(key, ttl, data)
    else: _mem_cache[key] = data

# ── Load models ────────────────────────────────────────────────────────────────
import xgboost as xgb
import shap

MODEL_DIR    = "models"
xgb_model    = pickle.load(open(f"{MODEL_DIR}/xgb_model.pkl",    "rb"))
causal_state = json.load(open(f"{MODEL_DIR}/causal_state.json"))
category_map = json.load(open(f"{MODEL_DIR}/category_map.json"))
model_meta   = json.load(open(f"{MODEL_DIR}/model_meta.json"))
bandit_state = json.load(open(f"{MODEL_DIR}/bandit_state.json"))

FEATURES = model_meta["features"]
MAE      = model_meta["mae"]
# ── SHAP — 3-tier fallback (fixes '[5E-1]' XGBoost version-compat bug) ─────────
#
#  Root cause: pkl saved with XGBoost 1.7.6.  Newer runtimes render binary-split
#  thresholds (e.g. 0.5) as '[5E-1]' in get_dump() text format.  SHAP's parser
#  calls float('[5E-1]') → ValueError.  Even loading xgb_model.json (also 1.7.6)
#  does NOT fix this — SHAP still calls get_dump() on the resulting Booster.
#
#  Tier 1 (fastest): pass the sklearn XGBRegressor wrapper directly — SHAP uses
#           a different internal code-path that avoids text-dump parsing.
#  Tier 2 (fallback): re-serialise booster as .ubj (current-runtime binary
#           format), reload, pass fresh Booster — get_dump() output matches
#           current SHAP parser expectations.
#  Tier 3 (last resort): shap.Explainer auto-detects model type; uses a
#           background dataset to avoid the text-parse path entirely.

import tempfile as _tf

_shap_explainer = None
SHAP_ON = False

# Tier 1 — sklearn wrapper
try:
    _shap_explainer = shap.TreeExplainer(xgb_model)
    SHAP_ON = True
    print("[OK] SHAP ready (tier-1: sklearn wrapper)")
except Exception as _e1:
    print(f"[SHAP tier-1 failed] {_e1}")

    # Tier 2 — UBJ resave
    if not SHAP_ON:
        try:
            _t = _tf.NamedTemporaryFile(suffix=".ubj", delete=False, dir=MODEL_DIR)
            _t.close()
            xgb_model.get_booster().save_model(_t.name)
            _fresh_booster = xgb.Booster()
            _fresh_booster.load_model(_t.name)
            os.unlink(_t.name)
            _shap_explainer = shap.TreeExplainer(_fresh_booster)
            SHAP_ON = True
            print("[OK] SHAP ready (tier-2: ubj resave)")
        except Exception as _e2:
            print(f"[SHAP tier-2 failed] {_e2}")

    # Tier 3 — shap.Explainer with background data
    if not SHAP_ON:
        try:
            _bg = pd.DataFrame([{f: 0.0 for f in FEATURES}])
            _shap_explainer = shap.Explainer(xgb_model, _bg)
            SHAP_ON = True
            print("[OK] SHAP ready (tier-3: auto Explainer)")
        except Exception as _e3:
            print(f"[WARN] SHAP disabled (all tiers failed): {_e3}")

print(f"Models loaded | MAE: ₹{MAE:.2f} | Redis: {REDIS_ON} | SHAP: {SHAP_ON}")

# ── Flipkart category → training category mapping ─────────────────────────────
CATEGORY_MAP_FK = {
    "mobiles":     "mobiles_&_accessories",
    "laptops":     "computers",
    "televisions": "televisions_&_video",
    "audio":       "audio",
    "tablets":     "computers",
    "smartwatch":  "wearable_smart_devices",
    "cameras":     "cameras_&_accessories",
    "electronics": "electronics",
}

# ── Feature engineering ────────────────────────────────────────────────────────
def engineer_row(comp_1, comp_2, comp_3, qty, customers,
                 freight_price, product_score, product_weight_g,
                 product_photos_qty, weekend, holiday,
                 total_price, category_name,
                 retail_price=None, market_price=None):

    comp_avg = (comp_1 + comp_2 + comp_3) / 3
    comp_min = min(comp_1, comp_2, comp_3)
    comp_max = max(comp_1, comp_2, comp_3)

    price = market_price or comp_avg
    mrp   = retail_price or price * 1.2

    train_cat    = CATEGORY_MAP_FK.get(category_name, "electronics")
    cat_enc      = category_map.get(train_cat, 0)
    discount_pct = (mrp - price) / (mrp + 1e-9)

    row = {
        "comp_avg_price":   comp_avg,
        "comp_min_price":   comp_min,
        "comp_max_price":   comp_max,
        "price_vs_comp":    price / (comp_avg + 1e-9),
        "price_gap":        price - comp_min,
        "discount_pct":     discount_pct,
        "log_retail":       np.log1p(mrp),
        "price_per_rating": price / (product_score + 0.1),
        "margin_proxy":     price / (mrp + 1e-9),
        "comp_spread":      comp_max - comp_min,
        "above_comp_avg":   int(price > comp_avg),
        "rating":           product_score,
        "brand_encoded":    0,
        "category_encoded": cat_enc,
    }
    return pd.DataFrame([row])[FEATURES]

# ── Constraints engine ─────────────────────────────────────────────────────────
MARGIN_FLOOR = 1.10
COMP_CEILING = 1.15
MAX_JUMP     = 0.25

def apply_constraints(price, freight, comp_min, market_price, mrp=None):
    original = price
    price = max(price, freight * MARGIN_FLOOR)
    price = min(price, comp_min * COMP_CEILING)
    price = min(price, market_price * (1 + MAX_JUMP))
    price = max(price, market_price * (1 - MAX_JUMP))
    if mrp and mrp > 0:
        price = min(price, mrp)
    return round(price, 2), price != original

# ── Causal price ───────────────────────────────────────────────────────────────
def causal_optimal_price(comp_avg):
    e  = causal_state["elasticity"]
    bp = causal_state["base_price"]
    bq = causal_state["base_qty"]
    opt     = (bq - e * bp) / (-2 * e + 1e-9)
    blended = 0.5 * opt + 0.5 * comp_avg
    return round(blended, 2), round(blended - 1.96*MAE, 2), round(blended + 1.96*MAE, 2)

# ── Bandit ─────────────────────────────────────────────────────────────────────
def best_bandit_multiplier():
    arms = bandit_state["arms"]
    best = max(arms, key=lambda a: arms[a]["avg_reward"])
    return float(best), arms[best]["avg_reward"], arms[best]["n_pulls"]

# ── SHAP explanation — handles TreeExplainer (ndarray) + shap.Explainer (Explanation obj)
def get_shap_explanation(X_row):
    if not SHAP_ON or _shap_explainer is None:
        return []
    try:
        result = _shap_explainer(X_row) if isinstance(_shap_explainer, shap.Explainer) \
                 else _shap_explainer.shap_values(X_row)
        # Unify: extract ndarray of shape (n_features,)
        if hasattr(result, "values"):          # shap.Explanation object
            vals = np.array(result.values[0])
        else:                                  # plain ndarray from TreeExplainer
            vals = np.array(result[0])
        top3 = sorted(zip(FEATURES, vals), key=lambda x: abs(x[1]), reverse=True)[:3]
        return [{"feature": f, "effect": round(float(v), 2)} for f, v in top3]
    except Exception as e:
        print(f"[SHAP] {e}")
        return []

# ── Core prediction ────────────────────────────────────────────────────────────
def run_prediction(comp_1, comp_2, comp_3, qty, customers,
                   freight_price, product_score, product_weight_g,
                   product_photos_qty, weekend, holiday, total_price,
                   category_name, product_name, market_price,
                   mrp=None, use_cache=True):

    comp_avg = (comp_1 + comp_2 + comp_3) / 3
    comp_min = min(comp_1, comp_2, comp_3)

    if use_cache:
        ck     = "price:" + hashlib.md5(f"{comp_1}{comp_2}{comp_3}{category_name}".encode()).hexdigest()[:12]
        cached = cache_get(ck)
        if cached:
            cached["source"] = "cache"
            return cached

    t0 = time.time()

    X = engineer_row(
        comp_1=comp_1, comp_2=comp_2, comp_3=comp_3,
        qty=qty, customers=customers, freight_price=freight_price,
        product_score=product_score, product_weight_g=product_weight_g,
        product_photos_qty=product_photos_qty, weekend=weekend, holiday=holiday,
        total_price=total_price, category_name=category_name,
        retail_price=mrp, market_price=market_price,
    )
    xgb_price = float(xgb_model.predict(X)[0])

    causal_price, ci_low, ci_high = causal_optimal_price(comp_avg)

    constrained, constraint_hit = apply_constraints(
        causal_price, freight_price, comp_min, market_price, mrp)

    multiplier, arm_reward, arm_pulls = best_bandit_multiplier()
    bandit_price = round(constrained * multiplier, 2)

    if mrp and mrp > 0:
        bandit_price = min(bandit_price, mrp)
        ci_high      = min(ci_high, mrp)

    explanation  = get_shap_explanation(X)
    margin_pct   = round((bandit_price - freight_price) / bandit_price * 100, 1)
    vs_comp_pct  = round((bandit_price - comp_avg) / comp_avg * 100, 1)
    vs_market    = round((bandit_price - market_price) / market_price * 100, 1)
    strategy     = "UNDERCUT" if bandit_price < comp_avg else "PREMIUM"

    result = {
        "product_name":      product_name,
        "category":          category_name,
        "market_price":      round(market_price, 2),
        "mrp":               round(mrp, 2) if mrp else None,
        "xgb_price":         round(xgb_price, 2),
        "causal_price":      causal_price,
        "constrained_price": constrained,
        "bandit_price":      bandit_price,
        "ci_low":            ci_low,
        "ci_high":           ci_high,
        "competitor_avg":    round(comp_avg, 2),
        "competitor_min":    round(comp_min, 2),
        "comp_1":            round(comp_1, 2),
        "comp_2":            round(comp_2, 2),
        "comp_3":            round(comp_3, 2),
        "margin_pct":        margin_pct,
        "vs_competitor_pct": vs_comp_pct,
        "vs_market_pct":     vs_market,
        "strategy":          strategy,
        "constraint_hit":    constraint_hit,
        "bandit_multiplier": multiplier,
        "bandit_arm_reward": round(arm_reward, 4),
        "explanation":       explanation,
        "source":            "model",
        "latency_ms":        round((time.time() - t0) * 1000, 1),
        "timestamp":         datetime.now().isoformat(),
    }

    if use_cache:
        cache_set(ck, result, ttl=60)
    return result

# ── Request schema ─────────────────────────────────────────────────────────────
class PricingRequest(BaseModel):
    product_name:       str
    market_price:       float
    comp_1:             float
    comp_2:             float
    comp_3:             float
    category_name:      str
    mrp:                Optional[float] = None
    qty:                Optional[float] = 50.0
    customers:          Optional[float] = 30.0
    freight_price:      Optional[float] = 100.0
    product_score:      Optional[float] = 4.0
    product_weight_g:   Optional[float] = 800.0
    product_photos_qty: Optional[float] = 4.0
    weekend:            Optional[int]   = 0
    holiday:            Optional[int]   = 0
    total_price:        Optional[float] = None

# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/")
def health():
    return {"status": "running", "redis": REDIS_ON, "shap": SHAP_ON, "source": "Flipkart"}

@app.post("/predict")
def predict(req: PricingRequest):
    total = req.total_price or (req.market_price * req.qty)
    return run_prediction(
        comp_1=req.comp_1, comp_2=req.comp_2, comp_3=req.comp_3,
        qty=req.qty, customers=req.customers, freight_price=req.freight_price,
        product_score=req.product_score, product_weight_g=req.product_weight_g,
        product_photos_qty=req.product_photos_qty, weekend=req.weekend,
        holiday=req.holiday, total_price=total, category_name=req.category_name,
        product_name=req.product_name, market_price=req.market_price, mrp=req.mrp,
    )

@app.get("/products")
def get_products():
    p = cache_get("live_products")
    return p if p else {"error": "No live products. Run: python scraper.py"}

@app.get("/products/{category}")
def get_product(category: str):
    p = cache_get("live_products") or {}
    return p.get(category, {"error": f"Category '{category}' not cached"})

# ── Category product LIST — all products in category (for browse grid) ─────────
@app.get("/category/{cat}/products")
def get_category_list(cat: str):
    from scraper import get_category_product_list, fetch_category_products, CATEGORIES
    products = get_category_product_list(cat)
    if not products:
        cat_id = CATEGORIES.get(cat)
        if cat_id:
            products = fetch_category_products(cat_id, n=10)
            if products:
                cache_set(f"cat_list:{cat}", products, ttl=360)
    return products or []

# ── Search results LIST — any query, returns multiple products ─────────────────
@app.get("/search/results")
def search_results_list(q: str, n: int = 12):
    from scraper import search_flipkart_products
    if not q or len(q.strip()) < 2:
        return {"error": "Query too short"}
    products = search_flipkart_products(q.strip(), n=n)
    if products:
        cache_set(f"search_list:{q.lower().strip()}", products, ttl=300)
    return products or []


# ── Search ─────────────────────────────────────────────────────────────────────
@app.get("/search")
def search_product(q: str):
    from scraper import search_and_cache
    if not q or len(q.strip()) < 2:
        return {"error": "Query too short"}
    prod = search_and_cache(q.strip())
    if not prod:
        return {"error": f"No results for '{q}'. Try a different query."}
    res = run_prediction(
        comp_1=prod["comp_1"], comp_2=prod["comp_2"], comp_3=prod["comp_3"],
        qty=50, customers=30, freight_price=100,
        product_score=prod.get("rating", 4.0),
        product_weight_g=500, product_photos_qty=4,
        weekend=0, holiday=0,
        total_price=prod["market_price"] * 50,
        category_name=prod.get("category", "electronics"),
        product_name=prod["product_name"],
        market_price=prod["market_price"],
        mrp=prod.get("mrp"),
        use_cache=False,
    )
    return {"product": prod, "pricing": res}

# ── Suggestions endpoint — returns top N tracked products ──────────────────────
@app.get("/suggestions")
def get_suggestions(n: int = 6):
    """Return top cached products as deal suggestions for dashboard landing page."""
    products = cache_get("live_products") or {}
    suggestions = []
    for cat, prod in products.items():
        suggestions.append({
            "category":     cat,
            "brand":        prod.get("brand", "—"),
            "product_name": prod["product_name"][:55],
            "market_price": prod["market_price"],
            "mrp":          prod.get("mrp", prod["market_price"]),
            "discount_pct": prod.get("discount_pct", 0),
            "rating":       prod.get("rating", 4.0),
            "image":        prod.get("image", ""),
            "product_url":  prod.get("product_url", ""),
            "comp_avg":     prod.get("comp_avg", prod["market_price"]),
            "pid":          prod.get("pid", cat),
        })
    # Sort by highest discount
    suggestions.sort(key=lambda x: x["discount_pct"], reverse=True)
    return suggestions[:n]

@app.post("/feedback")
def receive_feedback(arm: float, actual_revenue: float):
    arm_str = str(arm)
    if arm_str in bandit_state["arms"]:
        s = bandit_state["arms"][arm_str]
        n = s["n_pulls"] + 1
        s["avg_reward"] = (s["avg_reward"] * (n-1) + actual_revenue) / n
        s["n_pulls"] = n
        with open(f"{MODEL_DIR}/bandit_state.json", "w") as f:
            json.dump(bandit_state, f)
        return {"status": "updated", "arm": arm}
    return {"error": "arm not found"}

@app.get("/drift")
def drift_status():
    try:
        report = json.load(open(f"{MODEL_DIR}/drift_report.json"))
        drift  = report["metrics"][0]["result"]["dataset_drift"]
        return {"drift_detected": drift, "status": "alert" if drift else "ok"}
    except:
        return {"drift_detected": False, "status": "no report yet"}

# ── WebSocket ──────────────────────────────────────────────────────────────────
class ConnectionManager:
    def __init__(self): self.active: list[WebSocket] = []
    async def connect(self, ws):
        await ws.accept(); self.active.append(ws)
    def disconnect(self, ws):
        if ws in self.active: self.active.remove(ws)
    async def broadcast(self, msg):
        dead = []
        for ws in self.active:
            try: await ws.send_json(msg)
            except: dead.append(ws)
        for ws in dead: self.disconnect(ws)

manager = ConnectionManager()

async def broadcast_live_prices():
    while True:
        products = cache_get("live_products")
        if products and manager.active:
            for cat, prod in list(products.items())[:3]:
                try:
                    res = run_prediction(
                        comp_1=prod["comp_1"], comp_2=prod["comp_2"],
                        comp_3=prod["comp_3"], qty=50, customers=30,
                        freight_price=100, product_score=prod.get("rating", 4.0),
                        product_weight_g=500, product_photos_qty=4,
                        weekend=0, holiday=0,
                        total_price=prod["market_price"] * 50,
                        category_name=cat, product_name=prod["product_name"],
                        market_price=prod["market_price"], mrp=prod.get("mrp"),
                        use_cache=False,
                    )
                    await manager.broadcast({
                        "type": "price_update", "category": cat,
                        **{k: res[k] for k in [
                            "product_name", "bandit_price", "causal_price",
                            "ci_low", "ci_high", "competitor_avg",
                            "strategy", "vs_competitor_pct", "timestamp"]}
                    })
                    await asyncio.sleep(1)
                except Exception as e:
                    print(f"WS error: {e}")
        await asyncio.sleep(10)

@app.on_event("startup")
async def startup():
    asyncio.create_task(broadcast_live_prices())

@app.websocket("/ws/prices")
async def ws_prices(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True: await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# ── SKU history ────────────────────────────────────────────────────────────────
from scraper import get_sku_history, update_sku_history

class HistoryUpdate(BaseModel):
    market_price:  float
    comp_avg:      float
    mrp:           Optional[float] = None
    optimal_price: Optional[float] = None

@app.get("/history/{pid}")
def get_history(pid: str):
    return get_sku_history(pid)

@app.post("/history/{pid}/update")
def push_history(pid: str, body: HistoryUpdate):
    history = update_sku_history(
        pid=pid, market_price=body.market_price,
        comp_avg=body.comp_avg, mrp=body.mrp or body.market_price,
        optimal_price=body.optimal_price,
    )
    return {"pid": pid, "points": len(history)}