# dashboard.py — Real-Time Causal Pricing Engine (Flipkart)
# Run: streamlit run dashboard.py

import streamlit as st
import requests, json, time
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(page_title="Causal Pricing Engine", page_icon="🛒", layout="wide")

API = "http://localhost:8000"

CAT_ICONS = {
    "mobiles":     "📱",
    "laptops":     "💻",
    "televisions": "📺",
    "audio":       "🎧",
    "tablets":     "📲",
    "smartwatch":  "⌚",
    "cameras":     "📷",
    "electronics": "🔌",
}
CATEGORIES = list(CAT_ICONS.keys())

# ── Session state ──────────────────────────────────────────────────────────────
_defaults = {
    "mode":            "home",
    "browse_cat":      None,
    "browse_products": [],
    "search_query":    "",
    "search_products": [],
    "active_product":  None,
    "active_pricing":  None,
    "active_source":   None,
    "suggestions":     [],
    "all_products":    {},
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Helpers ────────────────────────────────────────────────────────────────────
def api_get(path, params=None, timeout=8):
    try:    return requests.get(f"{API}{path}", params=params, timeout=timeout).json()
    except: return None

def api_post(path, body, timeout=8):
    try:    return requests.post(f"{API}{path}", json=body, timeout=timeout).json()
    except: return None

def api_alive():
    try:    return requests.get(f"{API}/", timeout=2).status_code == 200
    except: return False

def inr(val):
    try:    return f"₹{float(val):,.0f}"
    except: return "₹—"

def fetch_live_products():
    r = api_get("/products")
    return r if isinstance(r, dict) and "error" not in r else {}

def fetch_suggestions():
    r = api_get("/suggestions", {"n": 8})
    return r if isinstance(r, list) else []

def fetch_category_products(cat):
    r = api_get(f"/category/{cat}/products", timeout=15)
    return r if isinstance(r, list) else []

def fetch_search_results(query):
    r = api_get("/search/results", {"q": query, "n": 12}, timeout=20)
    return r if isinstance(r, list) else []

def fetch_pricing(prod, cat="electronics", freight=100, score=4.0):
    return api_post("/predict", {
        "product_name":  prod.get("product_name", prod.get("name","—")),
        "market_price":  prod["market_price"],
        "mrp":           prod.get("mrp"),
        "comp_1":        prod["comp_1"],
        "comp_2":        prod["comp_2"],
        "comp_3":        prod["comp_3"],
        "category_name": cat,
        "freight_price": freight,
        "product_score": score,
        "qty":           50,
        "customers":     30,
    })

def fetch_sku_history(pid):
    r = api_get(f"/history/{pid}")
    return r if isinstance(r, list) else []

def detect_category(query):
    q = query.lower()
    if any(w in q for w in ["mobile","phone","smartphone","iphone","samsung","oneplus","redmi","realme","vivo","oppo"]): return "mobiles"
    if any(w in q for w in ["laptop","notebook","macbook","chromebook"]): return "laptops"
    if any(w in q for w in ["tv","television","qled","oled"]): return "televisions"
    if any(w in q for w in ["headphone","earphone","speaker","audio","boat","tws"]): return "audio"
    if any(w in q for w in ["tablet","ipad"]): return "tablets"
    if any(w in q for w in ["watch","smartwatch"]): return "smartwatch"
    if any(w in q for w in ["camera","dslr","mirrorless"]): return "cameras"
    return "electronics"

def enrich_product(raw, siblings, cat):
    comps = [p for p in siblings if p.get("pid") != raw.get("pid")]
    while len(comps) < 3: comps.append(raw)
    comps = comps[:3]
    name  = raw.get("name", raw.get("product_name","Unknown"))
    return {
        "pid":          raw.get("pid",""),
        "product_name": name,
        "brand":        raw.get("brand","—"),
        "product_url":  raw.get("url", raw.get("product_url","")),
        "image":        raw.get("image",""),
        "stock":        raw.get("stock","IN_STOCK"),
        "rating":       raw.get("rating",4.0),
        "rating_count": raw.get("rating_count",0),
        "category":     cat,
        "market_price": raw["price"],
        "mrp":          raw.get("mrp", raw["price"]),
        "discount_pct": raw.get("discount_pct",0),
        "comp_1":       comps[0]["price"],
        "comp_2":       comps[1]["price"],
        "comp_3":       comps[2]["price"],
        "comp_avg":     round(sum(c["price"] for c in comps)/3, 2),
        "comp_min":     round(min(c["price"] for c in comps), 2),
        "comp_max":     round(max(c["price"] for c in comps), 2),
        "competitors": [
            {"pid":c.get("pid",""),"name":c.get("name","")[:50],"brand":c.get("brand",""),
             "price":c["price"],"mrp":c.get("mrp",c["price"]),"rating":c.get("rating",4.0),
             "url":c.get("url","")}
            for c in comps
        ],
        "timestamp": raw.get("timestamp", datetime.now().isoformat()),
    }

def product_card_html(p):
    disc  = p.get("discount_pct",0)
    name  = p.get("name", p.get("product_name",""))[:55]
    brand = p.get("brand","—").upper()
    price = p.get("price", p.get("market_price",0))
    mrp   = p.get("mrp", price)
    badge = f'<div style="position:absolute;top:8px;right:8px;background:#E87722;color:#fff;font-size:0.68rem;font-weight:700;padding:2px 6px;border-radius:8px;">{disc:.0f}% OFF</div>' if disc >= 20 else ""
    return f"""
<div style="position:relative;border:1px solid #e8e8e8;border-radius:10px;
     padding:12px;background:#fff;min-height:200px;overflow:hidden;
     box-shadow:0 1px 3px rgba(0,0,0,0.06);margin-bottom:2px;">
  {badge}
  <div style="font-size:0.68rem;font-weight:700;color:#999;letter-spacing:.5px;">{brand}</div>
  <div style="font-size:0.78rem;color:#333;margin:3px 0 6px;min-height:44px;line-height:1.35;">{name}</div>
  <div style="font-size:1.1rem;font-weight:800;color:#1E3A5F;">{inr(price)}</div>
  <div style="font-size:0.72rem;color:#bbb;text-decoration:line-through;">{inr(mrp)}</div>
  <div style="font-size:0.75rem;color:#E87722;margin-top:2px;">⭐ {p.get('rating','—')} · {disc:.0f}% off</div>
  <div style="font-size:0.68rem;color:#ccc;margin-top:3px;">{p.get('stock','IN_STOCK')}</div>
</div>"""

# ── Pre-load ───────────────────────────────────────────────────────────────────
if not st.session_state.suggestions:
    st.session_state.suggestions  = fetch_suggestions()
if not st.session_state.all_products:
    st.session_state.all_products = fetch_live_products()

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🛒 Causal Pricing Engine")
    st.caption("Flipkart · XGBoost · DoWhy · UCB Bandit")
    st.divider()

    st.markdown("### 🔍 Search Flipkart")
    st.caption("Shirts, watches, mobiles — anything!")
    with st.form("sidebar_search_form", clear_on_submit=False):
        sq = st.text_input("Search products", placeholder="blue denim shirt, Nike shoes…",
                           label_visibility="collapsed", key="global_search")
        submitted_sb = st.form_submit_button("Search →", use_container_width=True, type="primary")
    if submitted_sb and sq.strip():
        st.session_state.search_query    = sq.strip()
        st.session_state.search_products = []
        st.session_state.mode            = "search_results"
        st.rerun()

    st.divider()
    st.markdown("### 📦 Browse Category")
    for cat in CATEGORIES:
        icon = CAT_ICONS[cat]
        is_active = (st.session_state.mode == "browse"
                     and st.session_state.browse_cat == cat)
        if st.button(f"{icon} {cat.title()}", key=f"sb_{cat}",
                     use_container_width=True,
                     type="primary" if is_active else "secondary"):
            st.session_state.mode            = "browse"
            st.session_state.browse_cat      = cat
            st.session_state.browse_products = []
            st.rerun()

    st.divider()
    if st.button("🏠 Home", use_container_width=True, key="sb_home"):
        st.session_state.mode = "home"; st.rerun()
    if st.button("🔄 Refresh Data", use_container_width=True, key="sb_ref"):
        st.session_state.suggestions  = fetch_suggestions()
        st.session_state.all_products = fetch_live_products()
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.mode == "home":
    st.markdown("# 🛒 Real-Time Causal Pricing Engine")
    st.markdown("*Flipkart live data · XGBoost + DoWhy causal inference · UCB Bandit*")
    if not api_alive():
        st.error("⚠️ API offline — run: `uvicorn api:app --reload --port 8000`")
    else:
        st.success("✅ API online")
    st.divider()

    st.markdown("## 🔥 Trending Deals")
    sugs = st.session_state.suggestions
    if not sugs:
        st.info("Run `python scraper.py` first, then click Refresh Data in sidebar.")
    else:
        cols = st.columns(4)
        for i, sg in enumerate(sugs[:8]):
            with cols[i % 4]:
                disc  = sg.get("discount_pct",0)
                icon  = CAT_ICONS.get(sg["category"],"🔹")
                color = "#27ae60" if disc>=40 else ("#e67e22" if disc>=20 else "#95a5a6")
                st.markdown(f"""
<div style="border:1px solid #e0e0e0;border-radius:10px;padding:12px;background:#fafafa;
     margin-bottom:4px;min-height:155px;">
  <div style="font-size:1.4rem;text-align:center">{icon}</div>
  <div style="font-weight:700;font-size:0.75rem;color:#555;margin:3px 0;">{sg['brand'].upper()}</div>
  <div style="font-size:0.73rem;color:#777;min-height:30px;">{sg['product_name'][:46]}</div>
  <div style="font-size:1.05rem;font-weight:800;color:#1E3A5F;margin-top:5px;">{inr(sg['market_price'])}</div>
  <div style="color:{color};font-size:0.78rem;font-weight:700;">{disc:.0f}% off · ⭐ {sg.get('rating','—')}</div>
</div>""", unsafe_allow_html=True)
                if st.button("📊 Analyse", key=f"home_an_{sg['pid']}",
                             use_container_width=True):
                    st.session_state.mode            = "browse"
                    st.session_state.browse_cat      = sg["category"]
                    st.session_state.browse_products = []
                    st.rerun()

    st.divider()
    st.markdown("## 📦 Browse by Category")
    cc = st.columns(4)
    for i, cat in enumerate(CATEGORIES):
        with cc[i % 4]:
            icon = CAT_ICONS[cat]
            p    = st.session_state.all_products.get(cat, {})
            st.markdown(f"""
<div style="border:1px solid #ddd;border-radius:10px;padding:14px;text-align:center;
     background:#fff;margin-bottom:4px;">
  <div style="font-size:2rem">{icon}</div>
  <div style="font-weight:700;font-size:0.9rem;margin:4px 0;">{cat.title()}</div>
  <div style="font-size:0.75rem;color:#888;">{p.get('brand','—') if p else '—'}</div>
  <div style="font-size:0.85rem;font-weight:700;color:#1E3A5F;">{inr(p.get('market_price',0)) if p else '—'}</div>
  <div style="font-size:0.72rem;color:#E87722;">{f"{p.get('discount_pct',0):.0f}% off" if p else ''}</div>
</div>""", unsafe_allow_html=True)
            if st.button(f"View {cat.title()}", key=f"home_cat_{cat}",
                         use_container_width=True):
                st.session_state.mode            = "browse"
                st.session_state.browse_cat      = cat
                st.session_state.browse_products = []
                st.rerun()

    st.divider()
    st.markdown("## 🔍 Quick Search — search ANYTHING")
    with st.form("home_search_form", clear_on_submit=False):
        hcols = st.columns([5, 1])
        with hcols[0]:
            hq = st.text_input("Quick search", placeholder="blue denim shirt, Nike shoes, Samsung TV…",
                               label_visibility="collapsed", key="home_q")
        with hcols[1]:
            submitted_hq = st.form_submit_button("Go →", use_container_width=True, type="primary")
    if submitted_hq and hq.strip():
        st.session_state.search_query    = hq.strip()
        st.session_state.search_products = []
        st.session_state.mode            = "search_results"
        st.rerun()
    st.caption("Not just electronics! Try: `blue denim jeans` · `Nike running shoes` · `gold earrings` · `pressure cooker`")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# BROWSE CATEGORY — full product grid
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.mode == "browse":
    cat  = st.session_state.browse_cat or CATEGORIES[0]
    icon = CAT_ICONS.get(cat,"📦")

    col_h1, col_h2 = st.columns([1,6])
    with col_h1:
        if st.button("← Home", key="browse_back"):
            st.session_state.mode = "home"; st.rerun()
    with col_h2:
        st.markdown(f"# {icon} {cat.title()} — All Products")

    st.divider()

    if not st.session_state.browse_products:
        with st.spinner(f"Loading {cat} products from Flipkart…"):
            st.session_state.browse_products = fetch_category_products(cat)

    prods = st.session_state.browse_products
    if not prods:
        st.warning(f"No {cat} products cached yet. Run `python scraper.py` and wait for the next scrape cycle (10 min).")
        st.stop()

    st.caption(f"**{len(prods)} products** found · Click **📊 Analyse Price** on any product to get optimal pricing")
    st.divider()

    cols = st.columns(4)
    for i, raw in enumerate(prods):
        enriched = enrich_product(raw, prods, cat)
        with cols[i % 4]:
            st.markdown(product_card_html(raw), unsafe_allow_html=True)
            if st.button("📊 Analyse Price", key=f"br_an_{i}_{raw.get('pid',i)}",
                         use_container_width=True):
                with st.spinner("Computing optimal price…"):
                    pricing = fetch_pricing(enriched, cat=cat,
                                            freight=100, score=float(raw.get("rating",4.0)))
                if pricing and "error" not in str(pricing):
                    st.session_state.active_product = enriched
                    st.session_state.active_pricing = pricing
                    st.session_state.active_source  = "browse"
                    st.session_state.mode           = "analyse"
                    st.rerun()
                else:
                    st.error("API not responding.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# SEARCH RESULTS — any query
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.mode == "search_results":
    query = st.session_state.search_query

    col_h1, col_h2 = st.columns([1,5])
    with col_h1:
        if st.button("← Home", key="sr_back"):
            st.session_state.mode = "home"; st.rerun()
    with col_h2:
        with st.form("sr_refine_form", clear_on_submit=False):
            srcols = st.columns([4, 1])
            with srcols[0]:
                nq = st.text_input("Refine search", value=query,
                                   label_visibility="collapsed", key="sr_q")
            with srcols[1]:
                submitted_sr = st.form_submit_button("🔍 Search", use_container_width=True)
        if submitted_sr and nq.strip():
            st.session_state.search_query    = nq.strip()
            st.session_state.search_products = []
            st.rerun()

    st.markdown(f"### 🔍 Results for: *{query}*")
    st.divider()

    if not st.session_state.search_products:
        with st.spinner(f"Searching Flipkart for **{query}**…"):
            st.session_state.search_products = fetch_search_results(query)

    results = st.session_state.search_products
    if not results:
        st.warning(
            f"No results found for **{query}**. "
            "The `/product-search` endpoint is disabled on the free RapidAPI plan. "
            "Upgrade your plan at rapidapi.com or browse categories instead."
        )
        st.stop()

    cat = detect_category(query)
    st.caption(f"**{len(results)} products** found · Category: `{cat}` · Click **📊 Analyse Price** on any")
    st.divider()

    cols = st.columns(4)
    for i, raw in enumerate(results):
        enriched = enrich_product(raw, results, cat)
        with cols[i % 4]:
            st.markdown(product_card_html(raw), unsafe_allow_html=True)
            if st.button("📊 Analyse Price", key=f"sr_an_{i}_{raw.get('pid',i)}",
                         use_container_width=True):
                with st.spinner("Computing optimal price…"):
                    pricing = fetch_pricing(enriched, cat=cat,
                                            freight=100, score=float(raw.get("rating",4.0)))
                if pricing and "error" not in str(pricing):
                    st.session_state.active_product = enriched
                    st.session_state.active_pricing = pricing
                    st.session_state.active_source  = "search"
                    st.session_state.mode           = "analyse"
                    st.rerun()
                else:
                    st.error("API not responding — is uvicorn running?")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# ANALYSE — pricing dashboard for selected product
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.mode == "analyse":
    prod = st.session_state.active_product
    res  = st.session_state.active_pricing

    if not prod or not res:
        st.error("No product selected.")
        if st.button("← Home"):
            st.session_state.mode = "home"; st.rerun()
        st.stop()

    source     = st.session_state.active_source
    back_mode  = "browse" if source == "browse" else "search_results"
    back_label = f"← Back to {st.session_state.browse_cat.title()} Products" \
                 if source == "browse" else f"← Back to '{st.session_state.search_query}'"
    if st.button(back_label, key="an_back"):
        st.session_state.mode = back_mode; st.rerun()

    sl1, sl2, sl3 = st.columns([2,2,1])
    with sl1: freight = st.slider("Freight (₹)", 50, 1000, 100, key="an_freight")
    with sl2: score   = st.slider("Product Score", 1.0, 5.0,
                                   float(prod.get("rating",4.0)), step=0.1, key="an_score")
    with sl3: auto    = st.toggle("🔄 Auto 5s", False, key="an_auto")

    cat = prod.get("category","electronics")
    pid = prod.get("pid","unknown")

    # Re-fetch if sliders changed from defaults
    if freight != 100 or score != float(prod.get("rating",4.0)):
        nr = fetch_pricing(prod, cat=cat, freight=freight, score=score)
        if nr and "error" not in str(nr):
            res = nr
            st.session_state.active_pricing = res

    try:
        requests.post(f"{API}/history/{pid}/update", timeout=2, json={
            "market_price":  prod["market_price"],
            "comp_avg":      prod["comp_avg"],
            "mrp":           prod.get("mrp"),
            "optimal_price": res["bandit_price"],
        })
    except: pass
    sku_history = fetch_sku_history(pid)

    icon = CAT_ICONS.get(cat,"📦")
    st.markdown(f"# {icon} Pricing Analysis")
    st.info(
        f"**{prod['product_name']}**  \n"
        f"Brand: `{prod.get('brand','—')}` | "
        f"Market: **{inr(prod['market_price'])}** | "
        f"MRP: **{inr(prod['mrp'])}** ({prod.get('discount_pct',0):.0f}% off) | "
        f"Comps: **{inr(prod['comp_1'])}** · **{inr(prod['comp_2'])}** · **{inr(prod['comp_3'])}**"
    )
    if prod.get("product_url"):
        st.link_button("🔗 View on Flipkart", prod["product_url"])
    st.divider()

    m1,m2,m3,m4,m5,m6,m7 = st.columns(7)
    m1.metric("🎯 Optimal",     inr(res['bandit_price']),   f"{res['vs_market_pct']:+.1f}% vs market")
    m2.metric("📊 95% CI",      f"{inr(res['ci_low'])} – {inr(res['ci_high'])}")
    m3.metric("🏪 Comp Avg",    inr(res['competitor_avg']), f"{res['vs_competitor_pct']:+.1f}%")
    m4.metric("💰 Margin",      f"{res['margin_pct']:.1f}%")
    m5.metric("🎲 Bandit Mult", f"{res['bandit_multiplier']:.2f}x")
    m6.metric("⚡ Strategy",    res["strategy"])
    m7.metric("🤖 XGB Raw",     inr(res['xgb_price']))

    diff      = abs(res['bandit_price'] - res['competitor_avg'])
    direction = "below" if res['bandit_price'] < res['competitor_avg'] else "above"
    why = f"Causal elasticity justifies pricing {inr(diff)} {direction} competitor average."
    if res.get("explanation"):
        top = res["explanation"][0]
        why += f" Top driver: **{top['feature']}** (effect: {top['effect']:+.2f})"
    if res.get("constraint_hit"):
        why += " ⚠️ Constraint applied."
    st.info(f"**Why this price?** {why}")
    st.divider()

    c1, c2 = st.columns([3,2])
    with c1:
        st.markdown("#### 📈 SKU Price History")
        if len(sku_history) >= 2:
            times = [h["time"]         for h in sku_history]
            mkt   = [h["market_price"] for h in sku_history]
            cavg  = [h["comp_avg"]     for h in sku_history]
            mrp_h = [h.get("mrp", h["market_price"]) for h in sku_history]
            opt   = [h["optimal_price"] for h in sku_history if h.get("optimal_price")]
            opt_t = [sku_history[i]["time"] for i,h in enumerate(sku_history) if h.get("optimal_price")]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=times, y=mrp_h, name="MRP",
                line=dict(color="#ccc", width=1, dash="dot")))
            fig.add_trace(go.Scatter(x=times, y=cavg, name="Comp Avg",
                line=dict(color="#E87722", width=1.5, dash="dash")))
            fig.add_trace(go.Scatter(x=times, y=mkt, name="Market",
                line=dict(color="#888", width=1.5), marker=dict(size=5)))
            if opt:
                fig.add_trace(go.Scatter(x=opt_t, y=opt, name="Optimal",
                    line=dict(color="#1E3A5F", width=2.5), marker=dict(size=7, symbol="star")))
            fig.update_layout(height=300, margin=dict(l=40,r=20,t=20,b=40),
                legend=dict(orientation="h", y=-0.3),
                yaxis_title="Price (₹)", xaxis_title="Time")
            st.plotly_chart(fig)
        else:
            st.caption(f"Building history — {len(sku_history)}/2 data points so far")

    with c2:
        st.markdown("#### 🏪 Competitor Breakdown")
        comp_names  = (["Our Optimal"] + [
            c.get("brand",f"Comp {i+1}")[:15]
            for i,c in enumerate(prod.get("competitors",[]))
        ])[:4]
        comp_prices = ([res["bandit_price"]] +
                       [prod["comp_1"], prod["comp_2"], prod["comp_3"]])[:4]
        fig2 = go.Figure(go.Bar(
            x=comp_prices, y=comp_names, orientation="h",
            marker_color=["#1E3A5F"]+["#E87722"]*3,
            text=[inr(p) for p in comp_prices], textposition="outside",
        ))
        if prod.get("mrp"):
            fig2.add_vline(x=prod["mrp"], line_color="#ccc", line_dash="dot",
                           annotation_text=f"MRP {inr(prod['mrp'])}")
        fig2.update_layout(height=300, margin=dict(l=20,r=80,t=20,b=40),
            xaxis_title="Price (₹)")
        st.plotly_chart(fig2)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("#### 📉 Causal Price Elasticity")
        try:
            import os
            cs = json.load(open("models/causal_state.json")) \
                 if os.path.exists("models/causal_state.json") \
                 else {"elasticity":-0.5,"base_price":res["competitor_avg"],"base_qty":50}
            e, bp, bq = cs["elasticity"], cs["base_price"], cs["base_qty"]
            px_arr  = np.linspace(res["competitor_avg"]*0.6, res["competitor_avg"]*1.5, 100)
            qty_arr = bq + e*(px_arr - bp)
            rev_arr = px_arr * np.clip(qty_arr, 0, None)
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=px_arr, y=rev_arr, name="Revenue",
                line=dict(color="#1E3A5F", width=2)))
            fig3.add_vline(x=res["bandit_price"], line_color="#1E3A5F",
                annotation_text=f"Optimal {inr(res['bandit_price'])}", annotation_position="top right")
            fig3.add_vline(x=res["competitor_avg"], line_color="#E87722", line_dash="dash",
                annotation_text=f"Comp avg {inr(res['competitor_avg'])}", annotation_position="top left")
            if res.get("mrp"):
                fig3.add_vline(x=res["mrp"], line_color="#ccc", line_dash="dot",
                    annotation_text=f"MRP {inr(res['mrp'])}")
            fig3.add_vrect(x0=res["ci_low"], x1=res["ci_high"],
                fillcolor="rgba(30,58,95,0.07)", line_width=0)
            fig3.update_layout(height=280, margin=dict(l=40,r=20,t=20,b=40),
                xaxis_title="Price (₹)", yaxis_title="Revenue (₹)")
            st.plotly_chart(fig3)
        except Exception as ex:
            st.caption(f"Chart error: {ex}")

    with c4:
        st.markdown("#### 🔍 SHAP Feature Impact")
        if res.get("explanation"):
            df_s = pd.DataFrame(res["explanation"])
            colors4 = ["#0F6E56" if v<0 else "#C0440E" for v in df_s["effect"]]
            fig4 = go.Figure(go.Bar(
                x=df_s["effect"], y=df_s["feature"], orientation="h",
                marker_color=colors4,
                text=[f"{v:+.2f}" for v in df_s["effect"]], textposition="outside",
            ))
            fig4.update_layout(height=280, margin=dict(l=20,r=60,t=20,b=40),
                xaxis_title="Price Impact", xaxis_zeroline=True)
            st.plotly_chart(fig4)
            st.caption("🟠 Raises price · 🟢 Lowers price")
        else:
            st.caption("SHAP not available.")

    st.divider()
    st.markdown("#### 🏪 Competitors vs Our Optimal Price")
    rows = []
    for c in prod.get("competitors",[]):
        diff = round(c["price"] - res["bandit_price"], 2)
        rows.append({
            "Brand":   c.get("brand","—")[:20],
            "Product": c.get("name","—")[:50],
            "Price":   inr(c["price"]),
            "MRP":     inr(c.get("mrp",c["price"])),
            "Rating":  f"{c.get('rating','—')} ⭐",
            "vs Ours": f"{'↑' if diff>0 else '↓'} {inr(abs(diff))}",
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), height=160)

    st.caption(f"Source: {res.get('source','model').upper()} | Latency: {res.get('latency_ms','—')}ms | Cat: {cat}")

    if auto:
        time.sleep(5)
        nr = fetch_pricing(prod, cat=cat, freight=freight, score=score)
        if nr: st.session_state.active_pricing = nr
        st.rerun()