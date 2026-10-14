from pathlib import Path
import warnings

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# OLIST E-COMMERCE PRODUCT & CUSTOMER ANALYTICS DASHBOARD
# Data source: Brazilian E-Commerce Public Dataset by Olist
# Expected folder: project_folder/data/
# ============================================================

st.set_page_config(
    page_title="Olist | Commerce Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------- DESIGN ----------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: #F5F7FB; color: #172033; }
    [data-testid="stSidebar"] { background: #101C32; }
    [data-testid="stSidebar"] * { color: #F4F7FC !important; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        padding: 8px 10px; border-radius: 9px; margin-bottom: 4px;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background: rgba(255,255,255,.08);
    }
    .brand-kicker { color: #9DB2D1; font-size: .72rem; letter-spacing: 2px;
        font-weight: 700; text-transform: uppercase; }
    .brand-title { color: white; font-family: 'Manrope', sans-serif;
        font-size: 1.35rem; font-weight: 800; margin: 4px 0 2px 0; }
    .brand-subtitle { color: #B9C7DB; font-size: .82rem; margin-bottom: 18px; }
    .page-eyebrow { color: #64748B; font-size: .72rem; font-weight: 700;
        letter-spacing: 1.6px; text-transform: uppercase; margin-bottom: 4px; }
    .page-title { color: #14213A; font-family: 'Manrope', sans-serif;
        font-size: 1.9rem; font-weight: 800; margin: 0; }
    .page-description { color: #64748B; font-size: .92rem; margin-top: 5px;
        margin-bottom: 18px; }
    .section-title { color: #172033; font-family: 'Manrope', sans-serif;
        font-size: 1.05rem; font-weight: 800; margin: 8px 0 12px 0; }
    .filter-heading { color: #64748B; font-size: .70rem; font-weight: 700;
        letter-spacing: 1.1px; margin-bottom: 7px; text-transform: uppercase; }
    div[data-baseweb="select"] > div, div[data-testid="stDateInput"] input {
        border-radius: 9px !important;
    }
    div[data-testid="stMetric"] {
        background: #FFFFFF; border: 1px solid #E8EDF5; border-radius: 14px;
        padding: 16px 18px; box-shadow: 0 3px 12px rgba(18, 38, 63, .035);
    }
    div[data-testid="stMetricLabel"] { color: #64748B; font-size: .8rem; }
    div[data-testid="stMetricValue"] { color: #14213A; font-family: 'Manrope', sans-serif;
        font-weight: 800; }
    div[data-testid="stPlotlyChart"] {
        background: white; border: 1px solid #E8EDF5; border-radius: 14px;
        padding: 8px; box-shadow: 0 3px 12px rgba(18, 38, 63, .025);
    }
    .insight-box { background: white; border: 1px solid #E8EDF5;
        border-left: 4px solid #16A6A1; border-radius: 10px;
        padding: 14px 16px; color: #334155; margin: 6px 0 12px 0; }
    .small-note { color: #64748B; font-size: .78rem; }
    hr { border-color: #E5EAF2; }
    </style>
    """,
    unsafe_allow_html=True,
)

NAVY = "#14213A"
TEAL = "#16A6A1"
BLUE = "#4776E6"
GOLD = "#E9AD55"
PALETTE = [TEAL, BLUE, GOLD, "#8B6FE8", "#E87979", "#5A9B79", "#7C8DA6"]
PLOT_LAYOUT = dict(
    paper_bgcolor="white",
    plot_bgcolor="white",
    font=dict(family="DM Sans, sans-serif", color="#475569", size=11),
    margin=dict(l=18, r=18, t=48, b=18),
    title_font=dict(family="Manrope, sans-serif", size=15, color=NAVY),
    legend_title_text="",
)

# ------------------------- DATA LOADING ----------------------
DATA_DIR = Path(__file__).resolve().parent / "data"


def find_csv(possible_names):
    """Find a CSV using a few common Olist filename variants."""
    for name in possible_names:
        candidate = DATA_DIR / name
        if candidate.exists():
            return candidate
    # Case-insensitive fallback
    if DATA_DIR.exists():
        lookup = {p.name.lower(): p for p in DATA_DIR.glob("*.csv")}
        for name in possible_names:
            if name.lower() in lookup:
                return lookup[name.lower()]
    return None


@st.cache_data(show_spinner="Loading and preparing the Olist dataset...")
def load_data():
    file_options = {
        "orders": ["orders.csv", "olist_orders_dataset.csv"],
        "customers": ["customers.csv", "olist_customers_dataset.csv"],
        "items": ["order_items.csv", "order_items_dataset.csv", "olist_order_items_dataset.csv"],
        "payments": ["payments.csv", "order_payments.csv", "olist_order_payments_dataset.csv"],
        "reviews": ["reviews.csv", "order_reviews.csv", "olist_order_reviews_dataset.csv"],
        "products": ["products.csv", "olist_products_dataset.csv"],
        "sellers": ["sellers.csv", "olist_sellers_dataset.csv"],
        "translation": ["product_category_name_translation.csv", "category_translation.csv"],
    }
    tables = {}
    missing = []
    for key, options in file_options.items():
        path = find_csv(options)
        if path is not None:
            tables[key] = pd.read_csv(path, low_memory=False)
        elif key not in ("sellers", "translation"):
            missing.append(f"{key}: {', '.join(options)}")

    if missing:
        raise FileNotFoundError(
            "Some required CSV files are missing from the data folder:\n- "
            + "\n- ".join(missing)
        )

    orders = tables["orders"].copy()
    customers = tables["customers"].copy()
    items = tables["items"].copy()
    payments = tables["payments"].copy()
    reviews = tables["reviews"].copy()
    products = tables["products"].copy()

    # Standardise common column names from Olist's original CSV files.
    date_col = "order_purchase_timestamp"
    if date_col not in orders.columns:
        raise ValueError("The orders CSV must contain 'order_purchase_timestamp'.")
    orders[date_col] = pd.to_datetime(orders[date_col], errors="coerce")

    for col in ["order_delivered_customer_date", "order_estimated_delivery_date",
                "order_approved_at", "order_delivered_carrier_date"]:
        if col in orders.columns:
            orders[col] = pd.to_datetime(orders[col], errors="coerce")

    if "customer_state" not in customers.columns:
        customers["customer_state"] = "Unknown"

    # Category translation is optional; use Portuguese names if unavailable.
    category_col = "product_category_name"
    if category_col not in products.columns:
        products[category_col] = "unknown"
    if "translation" in tables:
        translation = tables["translation"]
        if {"product_category_name", "product_category_name_english"}.issubset(translation.columns):
            products = products.merge(
                translation[["product_category_name", "product_category_name_english"]].drop_duplicates(),
                on="product_category_name", how="left"
            )
    if "product_category_name_english" not in products.columns:
        products["product_category_name_english"] = products["product_category_name"]
    products["category"] = (
        products["product_category_name_english"].fillna(products["product_category_name"])
        .fillna("Unknown").astype(str).str.replace("_", " ").str.title()
    )

    # One row per order, with customer state and dates.
    order_cols = ["order_id", "customer_id", "order_status", date_col]
    optional_order_cols = [c for c in [
        "order_delivered_customer_date", "order_estimated_delivery_date"
    ] if c in orders.columns]
    order_base = orders[order_cols + optional_order_cols].drop_duplicates("order_id")
    order_base = order_base.merge(
        customers[[c for c in ["customer_id", "customer_unique_id", "customer_state"] if c in customers.columns]]
        .drop_duplicates("customer_id"),
        on="customer_id", how="left"
    )

    # Item-level analysis: product price + freight and category.
    item_required = ["order_id", "product_id"]
    for col in item_required:
        if col not in items.columns:
            raise ValueError(f"The order items CSV must contain '{col}'.")
    if "price" not in items.columns:
        items["price"] = 0
    if "freight_value" not in items.columns:
        items["freight_value"] = 0
    items["price"] = pd.to_numeric(items["price"], errors="coerce").fillna(0)
    items["freight_value"] = pd.to_numeric(items["freight_value"], errors="coerce").fillna(0)
    product_info = products[[c for c in ["product_id", "category"] if c in products.columns]].drop_duplicates("product_id")
    item_analysis = items.merge(product_info, on="product_id", how="left")
    item_analysis["category"] = item_analysis["category"].fillna("Unknown")
    item_analysis["item_revenue"] = item_analysis["price"]
    item_analysis["item_total_with_freight"] = item_analysis["price"] + item_analysis["freight_value"]
    item_analysis = item_analysis.merge(
        order_base[["order_id", "customer_id", "customer_unique_id", "customer_state",
                    "order_status", date_col] + optional_order_cols],
        on="order_id", how="left"
    )

    # Payment amounts are summed at order level to prevent row duplication in order KPIs.
    if "payment_value" not in payments.columns:
        payments["payment_value"] = 0
    payments["payment_value"] = pd.to_numeric(payments["payment_value"], errors="coerce").fillna(0)
    payment_by_order = payments.groupby("order_id", as_index=False).agg(
        payment_value=("payment_value", "sum")
    )
    payment_methods = payments.copy()
    if "payment_type" not in payment_methods.columns:
        payment_methods["payment_type"] = "Unknown"

    # Review score averaged by order to avoid multiplying order rows.
    if "review_score" in reviews.columns:
        reviews["review_score"] = pd.to_numeric(reviews["review_score"], errors="coerce")
        review_by_order = reviews.groupby("order_id", as_index=False).agg(
            review_score=("review_score", "mean")
        )
    else:
        review_by_order = pd.DataFrame(columns=["order_id", "review_score"])

    orders_enriched = order_base.merge(payment_by_order, on="order_id", how="left")
    orders_enriched = orders_enriched.merge(review_by_order, on="order_id", how="left")
    orders_enriched["payment_value"] = orders_enriched["payment_value"].fillna(0)
    orders_enriched["purchase_date"] = orders_enriched[date_col].dt.date

    if {"order_delivered_customer_date", "order_estimated_delivery_date"}.issubset(orders_enriched.columns):
        orders_enriched["delivery_days"] = (
            orders_enriched["order_delivered_customer_date"] - orders_enriched[date_col]
        ).dt.total_seconds() / 86400
        orders_enriched["delivery_delay_days"] = (
            orders_enriched["order_delivered_customer_date"] - orders_enriched["order_estimated_delivery_date"]
        ).dt.total_seconds() / 86400
    else:
        orders_enriched["delivery_days"] = float("nan")
        orders_enriched["delivery_delay_days"] = float("nan")

    # Customer-level RFM table.
    customer_key = "customer_unique_id" if "customer_unique_id" in orders_enriched.columns else "customer_id"
    valid_orders = orders_enriched.dropna(subset=[date_col]).copy()
    if not valid_orders.empty:
        snapshot_date = valid_orders[date_col].max() + pd.Timedelta(days=1)
        rfm = valid_orders.groupby(customer_key, dropna=False).agg(
            last_purchase=(date_col, "max"),
            frequency=("order_id", "nunique"),
            monetary=("payment_value", "sum"),
        ).reset_index()
        rfm["recency_days"] = (snapshot_date - rfm["last_purchase"]).dt.days
        rfm["segment"] = "Regular"
        rfm.loc[(rfm["frequency"] >= 2) & (rfm["monetary"] >= rfm["monetary"].median()), "segment"] = "Loyal"
        rfm.loc[(rfm["frequency"] == 1) & (rfm["recency_days"] > rfm["recency_days"].median()), "segment"] = "At Risk"
        rfm.loc[(rfm["frequency"] == 1) & (rfm["recency_days"] <= rfm["recency_days"].median()), "segment"] = "Recent"
    else:
        rfm = pd.DataFrame(columns=[customer_key, "last_purchase", "frequency", "monetary", "recency_days", "segment"])

    return orders_enriched, item_analysis, payment_methods, reviews, rfm


def money(value):
    return f"R$ {value:,.0f}"


def chart_layout(fig, height=330):
    fig.update_layout(**PLOT_LAYOUT, height=height)
    fig.update_xaxes(showgrid=False, linecolor="#E8EDF5")
    fig.update_yaxes(gridcolor="#EEF2F7", zeroline=False)
    return fig


def page_header(eyebrow, title, description):
    st.markdown(f'<div class="page-eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.markdown(f'<h1 class="page-title">{title}</h1>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-description">{description}</div>', unsafe_allow_html=True)


# ------------------------- FILTERS ---------------------------
def show_filters(orders_enriched, item_analysis):
    st.markdown('<div class="filter-heading">REPORT FILTERS</div>', unsafe_allow_html=True)

    valid_dates = orders_enriched["order_purchase_timestamp"].dropna()
    if valid_dates.empty:
        min_date, max_date = pd.Timestamp("2016-01-01").date(), pd.Timestamp.today().date()
    else:
        min_date, max_date = valid_dates.min().date(), valid_dates.max().date()

    date_col, state_col, category_col = st.columns([1.5, 1, 1])
    with date_col:
        date_range = st.date_input(
            "Purchase date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="purchase_date_range",
        )
    # Only three states are offered: the three with the most orders in the loaded dataset.
    states_ranked = (
        orders_enriched.dropna(subset=["customer_state"])
        .groupby("customer_state")["order_id"].nunique()
        .sort_values(ascending=False)
    )
    top_states = states_ranked.head(3).index.tolist()
    if not top_states:
        top_states = sorted(orders_enriched["customer_state"].dropna().unique().tolist())[:3]

    # Only three categories are offered: the three with the highest item revenue.
    top_categories = (
        item_analysis.groupby("category")["item_revenue"].sum()
        .sort_values(ascending=False).head(3).index.tolist()
    )
    if not top_categories:
        top_categories = sorted(item_analysis["category"].dropna().unique().tolist())[:3]

    with state_col:
        selected_states = st.multiselect(
            "Top 3 states",
            options=top_states,
            default=top_states,
            key="dashboard_states",
            help="The three states with the highest order counts in the dataset.",
        )
    with category_col:
        selected_categories = st.multiselect(
            "Top 3 categories",
            options=top_categories,
            default=top_categories,
            key="dashboard_categories",
            help="The three product categories with the highest item revenue.",
        )

    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        start_date, end_date = date_range
    elif isinstance(date_range, (tuple, list)) and len(date_range) == 1:
        start_date, end_date = date_range[0], date_range[0]
    else:
        start_date, end_date = min_date, max_date

    st.caption(
        f"Showing data for **{start_date} – {end_date}** · "
        f"**{len(selected_states)} states** · **{len(selected_categories)} categories**. "
        "State/category lists are intentionally limited to three options each."
    )
    return start_date, end_date, selected_states, selected_categories


def apply_filters(orders_enriched, item_analysis, start_date, end_date, states, categories):
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)

    order_mask = orders_enriched["order_purchase_timestamp"].between(start_ts, end_ts)
    if states:
        order_mask &= orders_enriched["customer_state"].isin(states)
    else:
        order_mask &= False
    filtered_orders = orders_enriched.loc[order_mask].copy()
    filtered_order_ids = set(filtered_orders["order_id"].dropna().unique())

    item_mask = item_analysis["order_purchase_timestamp"].between(start_ts, end_ts)
    if states:
        item_mask &= item_analysis["customer_state"].isin(states)
    else:
        item_mask &= False
    if categories:
        item_mask &= item_analysis["category"].isin(categories)
    else:
        item_mask &= False
    filtered_items = item_analysis.loc[item_mask].copy()

    # Use order IDs from filtered orders for payment/review views; category filtering
    # applies only to item/category analytics.
    return filtered_orders, filtered_items, filtered_order_ids


# ------------------------- APP START -------------------------
try:
    orders, items, payment_methods, reviews, rfm = load_data()
except Exception as exc:
    st.error("The dashboard could not load the dataset.")
    st.markdown(
        "Place your Olist CSV files inside a folder named **data** beside `app.py`. "
        "Use the original Olist filenames or the shorter names listed below."
    )
    st.code(
        "project_folder/\n"
        "├── app.py\n"
        "└── data/\n"
        "    ├── orders.csv\n"
        "    ├── customers.csv\n"
        "    ├── order_items.csv\n"
        "    ├── payments.csv\n"
        "    ├── reviews.csv\n"
        "    ├── products.csv\n"
        "    ├── sellers.csv (optional)\n"
        "    └── product_category_name_translation.csv (optional)"
    )
    st.exception(exc)
    st.stop()

# Sidebar navigation
with st.sidebar:
    st.markdown(
        '<div class="brand-kicker">OLIST · BUSINESS INTELLIGENCE</div>'
        '<div class="brand-title">Commerce<br>Intelligence</div>'
        '<div class="brand-subtitle">Product · Customer · Operations</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    page = st.radio(
        "WORKSPACE",
        [
            "Executive Overview",
            "Product Analytics",
            "Customer Intelligence",
            "Orders & Payments",
            "Delivery & Reviews",
        ],
        label_visibility="visible",
    )
    st.markdown("---")
    st.markdown('<div class="small-note">DATASET</div>', unsafe_allow_html=True)
    st.markdown("Brazilian E-Commerce · Olist")
    st.markdown('<div class="small-note">Historical data: 2016–2018</div>', unsafe_allow_html=True)

# Shared report filters
start_date, end_date, selected_states, selected_categories = show_filters(orders, items)
filtered_orders, filtered_items, filtered_order_ids = apply_filters(
    orders, items, start_date, end_date, selected_states, selected_categories
)

# ------------------------- PAGE 1 ----------------------------
if page == "Executive Overview":
    page_header(
        "BUSINESS PERFORMANCE",
        "Executive Overview",
        "A concise view of sales, customer demand and service performance across the selected dataset slice.",
    )
    total_orders = filtered_orders["order_id"].nunique()
    customers_count = filtered_orders["customer_unique_id"].nunique() if "customer_unique_id" in filtered_orders.columns else filtered_orders["customer_id"].nunique()
    sales_value = filtered_items["item_revenue"].sum()
    aov = sales_value / total_orders if total_orders else 0
    avg_rating = filtered_orders["review_score"].mean() if not filtered_orders.empty else 0

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Orders", f"{total_orders:,}")
    k2.metric("Customers", f"{customers_count:,}")
    k3.metric("Item sales", money(sales_value))
    k4.metric("Sales / order", money(aov))
    k5.metric("Avg. review", f"{avg_rating:.2f} / 5" if pd.notna(avg_rating) else "—")

    st.markdown('<div class="section-title">Sales & demand</div>', unsafe_allow_html=True)
    left, right = st.columns([1.45, 1])
    with left:
        if not filtered_items.empty:
            monthly = filtered_items.dropna(subset=["order_purchase_timestamp"]).copy()
            monthly["month"] = monthly["order_purchase_timestamp"].dt.to_period("M").dt.to_timestamp()
            monthly_sales = monthly.groupby("month", as_index=False)["item_revenue"].sum()
            fig = px.area(monthly_sales, x="month", y="item_revenue", title="Monthly item sales",
                          labels={"month": "Month", "item_revenue": "Sales (R$)"})
            fig.update_traces(line_color=TEAL, fillcolor="rgba(22,166,161,0.14)")
            st.plotly_chart(chart_layout(fig, 340), use_container_width=True)
        else:
            st.info("No item sales match the selected filters.")
    with right:
        if not filtered_items.empty:
            cat_sales = filtered_items.groupby("category", as_index=False)["item_revenue"].sum().sort_values("item_revenue", ascending=True)
            fig = px.bar(cat_sales, x="item_revenue", y="category", orientation="h",
                         title="Sales by selected category", color="category",
                         color_discrete_sequence=PALETTE,
                         labels={"item_revenue": "Sales (R$)", "category": "Category"})
            fig.update_layout(showlegend=False)
            st.plotly_chart(chart_layout(fig, 340), use_container_width=True)
        else:
            st.info("No category data matches the selected filters.")

    st.markdown('<div class="section-title">Order health</div>', unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        if not filtered_orders.empty:
            status = filtered_orders["order_status"].fillna("Unknown").value_counts().rename_axis("status").reset_index(name="orders")
            fig = px.pie(status, names="status", values="orders", hole=.68, title="Order status mix",
                         color_discrete_sequence=PALETTE)
            fig.update_traces(textposition="inside", textinfo="percent")
            st.plotly_chart(chart_layout(fig, 320), use_container_width=True)
        else:
            st.info("No orders match the selected filters.")
    with right:
        if not filtered_orders.empty:
            daily = filtered_orders.dropna(subset=["order_purchase_timestamp"]).copy()
            daily["month"] = daily["order_purchase_timestamp"].dt.to_period("M").dt.to_timestamp()
            monthly_orders = daily.groupby("month", as_index=False)["order_id"].nunique()
            fig = px.line(monthly_orders, x="month", y="order_id", markers=True, title="Monthly order volume",
                          labels={"month": "Month", "order_id": "Orders"})
            fig.update_traces(line_color=BLUE, line_width=3)
            st.plotly_chart(chart_layout(fig, 320), use_container_width=True)

    st.markdown(
        '<div class="insight-box"><b>How to read this page:</b> Item sales use product item prices, '
        'excluding freight. Order counts are counted at unique order level. The dashboard uses historical '
        'Olist records, not live transactions.</div>',
        unsafe_allow_html=True,
    )

# ------------------------- PAGE 2 ----------------------------
elif page == "Product Analytics":
    page_header(
        "ASSORTMENT & REVENUE",
        "Product Analytics",
        "Compare category contribution, item pricing and the product mix within the selected states and dates.",
    )
    if filtered_items.empty:
        st.warning("No product records match the current filters. Select at least one state and category.")
    else:
        total_item_revenue = filtered_items["item_revenue"].sum()
        item_count = len(filtered_items)
        avg_item_price = filtered_items["price"].mean() if item_count else 0
        product_count = filtered_items["product_id"].nunique()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Item revenue", money(total_item_revenue))
        c2.metric("Items sold", f"{item_count:,}")
        c3.metric("Unique products", f"{product_count:,}")
        c4.metric("Average item price", money(avg_item_price))

        left, right = st.columns(2)
        with left:
            cat = filtered_items.groupby("category", as_index=False).agg(
                revenue=("item_revenue", "sum"), items=("order_id", "count")
            ).sort_values("revenue", ascending=False)
            fig = px.bar(cat.sort_values("revenue"), x="revenue", y="category", orientation="h",
                         title="Category revenue", color="revenue", color_continuous_scale=["#BCE9E5", TEAL],
                         labels={"revenue": "Revenue (R$)", "category": "Category"})
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(chart_layout(fig, 350), use_container_width=True)
        with right:
            fig = px.scatter(cat, x="items", y="revenue", size="revenue", color="category",
                             title="Item volume vs revenue",
                             labels={"items": "Items sold", "revenue": "Revenue (R$)"},
                             color_discrete_sequence=PALETTE)
            fig.update_layout(showlegend=False)
            st.plotly_chart(chart_layout(fig, 350), use_container_width=True)

        st.markdown('<div class="section-title">Category performance table</div>', unsafe_allow_html=True)
        table = filtered_items.groupby("category", as_index=False).agg(
            Items_Sold=("order_id", "count"),
            Unique_Products=("product_id", "nunique"),
            Revenue=("item_revenue", "sum"),
            Average_Price=("price", "mean"),
            Freight=("freight_value", "sum"),
        ).sort_values("Revenue", ascending=False)
        table["Revenue"] = table["Revenue"].round(2)
        table["Average_Price"] = table["Average_Price"].round(2)
        table["Freight"] = table["Freight"].round(2)
        st.dataframe(table, use_container_width=True, hide_index=True)

# ------------------------- PAGE 3 ----------------------------
elif page == "Customer Intelligence":
    page_header(
        "CUSTOMER VALUE & BEHAVIOUR",
        "Customer Intelligence",
        "Explore customer purchase frequency, value and simple RFM-style behavioural segments.",
    )
    filtered_customer_key = "customer_unique_id" if "customer_unique_id" in filtered_orders.columns else "customer_id"
    if filtered_orders.empty:
        st.warning("No customers match the current filters.")
    else:
        customer_summary = filtered_orders.groupby(filtered_customer_key, dropna=False).agg(
            Orders=("order_id", "nunique"),
            Spend=("payment_value", "sum"),
            Last_Purchase=("order_purchase_timestamp", "max"),
            Avg_Review=("review_score", "mean"),
        ).reset_index()
        repeat_customers = int((customer_summary["Orders"] > 1).sum())
        repeat_rate = repeat_customers / len(customer_summary) * 100 if len(customer_summary) else 0
        avg_spend = customer_summary["Spend"].mean() if len(customer_summary) else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Customers in slice", f"{len(customer_summary):,}")
        c2.metric("Repeat customers", f"{repeat_customers:,}")
        c3.metric("Repeat rate", f"{repeat_rate:.1f}%")
        c4.metric("Avg. customer spend", money(avg_spend))

        left, right = st.columns(2)
        with left:
            spend_dist = customer_summary.copy()
            spend_dist["Spend band"] = pd.cut(
                spend_dist["Spend"],
                bins=[-0.01, 100, 300, 700, float("inf")],
                labels=["Under R$100", "R$100–300", "R$300–700", "R$700+"],
            )
            spend_counts = spend_dist["Spend band"].value_counts(sort=False).rename_axis("Spend band").reset_index(name="Customers")
            fig = px.bar(spend_counts, x="Spend band", y="Customers", title="Customer spend distribution",
                         color="Spend band", color_discrete_sequence=PALETTE)
            fig.update_layout(showlegend=False)
            st.plotly_chart(chart_layout(fig, 340), use_container_width=True)
        with right:
            frequency = customer_summary["Orders"].value_counts().sort_index().head(8).rename_axis("Orders per customer").reset_index(name="Customers")
            fig = px.bar(frequency, x="Orders per customer", y="Customers", title="Purchase frequency",
                         color_discrete_sequence=[BLUE])
            st.plotly_chart(chart_layout(fig, 340), use_container_width=True)

        # Calculate RFM-style segments only for customers in the current filtered date/state slice.
        rfm_view = customer_summary.copy()
        snapshot = pd.to_datetime(rfm_view["Last_Purchase"], errors="coerce").max()
        rfm_view["Recency_Days"] = (snapshot - pd.to_datetime(rfm_view["Last_Purchase"], errors="coerce")).dt.days
        rfm_view["Segment"] = "Regular"
        if not rfm_view.empty:
            med_spend = rfm_view["Spend"].median()
            med_recency = rfm_view["Recency_Days"].median()
            rfm_view.loc[(rfm_view["Orders"] >= 2) & (rfm_view["Spend"] >= med_spend), "Segment"] = "Loyal"
            rfm_view.loc[(rfm_view["Orders"] == 1) & (rfm_view["Recency_Days"] > med_recency), "Segment"] = "At Risk"
            rfm_view.loc[(rfm_view["Orders"] == 1) & (rfm_view["Recency_Days"] <= med_recency), "Segment"] = "Recent"
        segments = rfm_view["Segment"].value_counts().rename_axis("Segment").reset_index(name="Customers")
        fig = px.pie(segments, names="Segment", values="Customers", hole=.62,
                     title="RFM-style customer segments", color_discrete_sequence=PALETTE)
        st.plotly_chart(chart_layout(fig, 330), use_container_width=True)
        st.caption("Segments are descriptive RFM-style rules for this dashboard, not a trained predictive model.")

        st.markdown('<div class="section-title">Customer summary</div>', unsafe_allow_html=True)
        display = customer_summary.rename(columns={filtered_customer_key: "Customer ID"})
        display["Spend"] = display["Spend"].round(2)
        display["Avg_Review"] = display["Avg_Review"].round(2)
        st.dataframe(display.sort_values("Spend", ascending=False).head(100), use_container_width=True, hide_index=True)
        st.caption("Table is limited to the top 100 customers by spend for readability.")

# ------------------------- PAGE 4 ----------------------------
elif page == "Orders & Payments":
    page_header(
        "TRANSACTION ANALYTICS",
        "Orders & Payments",
        "Review order statuses, payment methods and transaction value without double-counting split payments.",
    )
    if filtered_orders.empty:
        st.warning("No orders match the current filters.")
    else:
        total_payments = filtered_orders["payment_value"].sum()
        avg_payment = filtered_orders["payment_value"].mean()
        cancelled = filtered_orders["order_status"].astype(str).str.lower().eq("canceled").sum()
        cancel_rate = cancelled / len(filtered_orders) * 100 if len(filtered_orders) else 0
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Orders", f"{filtered_orders['order_id'].nunique():,}")
        c2.metric("Payment value", money(total_payments))
        c3.metric("Average payment / order", money(avg_payment))
        c4.metric("Cancelled order share", f"{cancel_rate:.1f}%")

        order_ids = set(filtered_orders["order_id"].dropna().unique())
        filtered_payment_rows = payment_methods[payment_methods["order_id"].isin(order_ids)].copy()
        left, right = st.columns(2)
        with left:
            if not filtered_payment_rows.empty:
                method = filtered_payment_rows.groupby("payment_type", as_index=False).agg(
                    Transactions=("order_id", "count"),
                    Value=("payment_value", "sum"),
                ).sort_values("Value", ascending=False)
                fig = px.bar(method, x="payment_type", y="Value", title="Payment value by method",
                             color="payment_type", color_discrete_sequence=PALETTE,
                             labels={"payment_type": "Payment method", "Value": "Value (R$)"})
                fig.update_layout(showlegend=False)
                st.plotly_chart(chart_layout(fig, 350), use_container_width=True)
            else:
                st.info("No payment rows available for the selected orders.")
        with right:
            status = filtered_orders["order_status"].fillna("Unknown").value_counts().rename_axis("Status").reset_index(name="Orders")
            fig = px.bar(status.sort_values("Orders"), x="Orders", y="Status", orientation="h",
                         title="Orders by status", color="Orders", color_continuous_scale=["#BCE9E5", TEAL])
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(chart_layout(fig, 350), use_container_width=True)

        st.markdown('<div class="section-title">Recent order records</div>', unsafe_allow_html=True)
        cols = [c for c in ["order_id", "customer_state", "order_status", "order_purchase_timestamp", "payment_value", "review_score"] if c in filtered_orders.columns]
        st.dataframe(
            filtered_orders[cols].sort_values("order_purchase_timestamp", ascending=False).head(100),
            use_container_width=True, hide_index=True
        )
        st.caption("Payment value is the sum of payment records belonging to each order.")

# ------------------------- PAGE 5 ----------------------------
elif page == "Delivery & Reviews":
    page_header(
        "FULFILMENT & CUSTOMER EXPERIENCE",
        "Delivery & Reviews",
        "Understand delivery duration, estimated-date delays and review scores for the selected order population.",
    )
    if filtered_orders.empty:
        st.warning("No orders match the current filters.")
    else:
        delivered = filtered_orders[
            filtered_orders["order_status"].astype(str).str.lower().eq("delivered")
        ].copy()
        delivery_days = delivered["delivery_days"].dropna()
        delay_days = delivered["delivery_delay_days"].dropna()
        avg_delivery = delivery_days.mean() if not delivery_days.empty else float("nan")
        late_rate = (delay_days > 0).mean() * 100 if not delay_days.empty else float("nan")
        avg_review = filtered_orders["review_score"].mean()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Delivered orders", f"{len(delivered):,}")
        c2.metric("Avg. delivery time", f"{avg_delivery:.1f} days" if pd.notna(avg_delivery) else "—")
        c3.metric("Delivered late", f"{late_rate:.1f}%" if pd.notna(late_rate) else "—")
        c4.metric("Average review", f"{avg_review:.2f} / 5" if pd.notna(avg_review) else "—")

        left, right = st.columns(2)
        with left:
            if not delivery_days.empty:
                delivery_df = pd.DataFrame({"Delivery days": delivery_days})
                fig = px.histogram(delivery_df, x="Delivery days", nbins=28,
                                   title="Distribution of delivery time",
                                   color_discrete_sequence=[TEAL])
                st.plotly_chart(chart_layout(fig, 350), use_container_width=True)
            else:
                st.info("Delivery dates are unavailable for the selected orders.")
        with right:
            score_data = filtered_orders["review_score"].dropna().value_counts().sort_index().rename_axis("Rating").reset_index(name="Reviews")
            if not score_data.empty:
                fig = px.bar(score_data, x="Rating", y="Reviews", title="Customer review score distribution",
                             color="Rating", color_continuous_scale=["#E87979", GOLD, TEAL])
                fig.update_layout(coloraxis_showscale=False)
                st.plotly_chart(chart_layout(fig, 350), use_container_width=True)
            else:
                st.info("No review scores are available for the selected orders.")

        if not delay_days.empty:
            late_summary = pd.DataFrame({
                "Delivery status": ["On or before estimate", "After estimate"],
                "Orders": [int((delay_days <= 0).sum()), int((delay_days > 0).sum())],
            })
            fig = px.pie(late_summary, names="Delivery status", values="Orders", hole=.65,
                         title="Delivery against estimated date",
                         color_discrete_sequence=[TEAL, GOLD])
            st.plotly_chart(chart_layout(fig, 310), use_container_width=True)

        st.markdown(
            '<div class="insight-box"><b>Metric definitions:</b> Delivery duration is the number of days '
            'from purchase to customer delivery. A late delivery is one delivered after the estimated date. '
            'Orders without the required timestamps are excluded from these delivery calculations.</div>',
            unsafe_allow_html=True,
        )

# ------------------------- FOOTER ----------------------------
st.markdown("---")
st.markdown(
    '<div class="small-note">OLIST COMMERCE INTELLIGENCE · Built with Streamlit, Pandas and Plotly · '
    'Source: Brazilian E-Commerce Public Dataset by Olist</div>',
    unsafe_allow_html=True,
)
