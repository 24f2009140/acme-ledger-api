
import os
import re
import json
import time
import logging
import threading
from datetime import date, datetime
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from flask import Flask, request, jsonify
from groq import Groq

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

CONTEXT_URL = os.environ["LEDGER_CONTEXT_URL"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
REFRESH_TOKEN = os.getenv("REFRESH_TOKEN", "")

PAGE_COUNT = 85
MAX_WORKERS = 16
TZ = ZoneInfo("Asia/Kolkata")

client = Groq(api_key=GROQ_API_KEY)
http = requests.Session()
http.headers.update({"Accept": "application/json"})

data_lock = threading.RLock()
refresh_lock = threading.Lock()

# The active snapshot is replaced only after a complete successful load.
snapshot = {
    "orders": pd.DataFrame(),
    "rates": {},
    "root": None,
    "loaded_at": None,
    "order_count": 0,
    "page_count": 0,
    "error": None,
}

QUERY_PROMPT = """
Convert the user's question into one JSON query plan.

Allowed metrics:
revenue, refund_count, refund_amount, order_count,
top_product, top_customer, top_region,
product_revenue, customer_spending, unknown.

Return this JSON shape:
{
  "metric": "revenue",
  "region": null,
  "product": null,
  "customer": null,
  "start_date": null,
  "end_date": null,
  "status": null,
  "limit": 1
}

Rules:
- Use ISO dates YYYY-MM-DD.
- Interpret month names as complete calendar months.
- Use inclusive start and end dates.
- "highest revenue product" means top_product.
- "highest spending customer" means top_customer.
- "how many refunded orders" means refund_count.
- "total refund value" means refund_amount.
- "total revenue" means revenue.
- "how many orders" means order_count.
- "revenue for a named product" means product_revenue.
- "spending by a named customer" means customer_spending.
- "top 5 products" means top_product with limit 5.
- Don't invent unspecified filters.
- Don't return SQL or executable code.
- Output JSON only.
"""


def validate_context_url():
    from urllib.parse import urlparse

    parsed = urlparse(CONTEXT_URL)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "exam.sanand.workers.dev"
        or not parsed.path.endswith("/questionData")
    ):
        raise RuntimeError("LEDGER_CONTEXT_URL is not the assigned questionData URL")


def get_api(path, page=None, timeout=15):
    """Call the assigned questionData proxy, preserving its signed parameters."""
    params = {"path": path}
    if page is not None:
        params["page"] = page

    response = http.get(CONTEXT_URL, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def records_from_response(value):
    """Extract a list of record dictionaries from common API envelopes."""
    if isinstance(value, list):
        if not value:
            return []

        if all(isinstance(item, dict) for item in value):
            return value

        for item in value:
            result = records_from_response(item)
            if result:
                return result

    elif isinstance(value, dict):
        for key in (
            "orders", "results", "records", "items",
            "data", "rows", "customers", "products", "refunds"
        ):
            if key in value:
                result = records_from_response(value[key])
                if result:
                    return result

        # Some endpoints return a single record.
        if any(key in value for key in ("id", "order_id", "orderId")):
            return [value]

    return []


def discover_rates(value):
    """Find the documented usd_per_unit mapping in the API root response."""
    if isinstance(value, dict):
        candidate = value.get("usd_per_unit")
        if isinstance(candidate, dict) and candidate:
            try:
                return {
                    str(currency).upper(): float(rate)
                    for currency, rate in candidate.items()
                }
            except (TypeError, ValueError):
                pass

        for child in value.values():
            result = discover_rates(child)
            if result:
                return result

    elif isinstance(value, list):
        for child in value:
            result = discover_rates(child)
            if result:
                return result

    return {}


def find_resource_paths(value):
    """Find resource paths mentioned in API documentation."""
    paths = set()

    def walk(node):
        if isinstance(node, dict):
            for key, child in node.items():
                if isinstance(child, str):
                    text = child
                    matches = re.findall(
                        r"""(?:path|endpoint|url|route)\s*["']?\s*[:=]\s*["'](/[^"' \t\r\n]+)""",
                        text,
                        flags=re.I,
                    )
                    paths.update(matches)

                    if child.startswith("/") and not child.startswith("//"):
                        paths.add(child.split("?")[0])

                walk(child)

        elif isinstance(node, list):
            for child in node:
                walk(child)

        elif isinstance(node, str):
            for match in re.findall(r"""["'](/(?:orders|refunds|products|customers|rates|exchange-rates)(?:/[^"']*)?)["']""", node, flags=re.I):
                paths.add(match.split("?")[0])

    walk(value)
    return paths


def load_orders():
    """Fetch all 85 pages. Fail the refresh if any page fails."""
    pages = {}
    failures = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(get_api, "/orders", page): page
            for page in range(1, PAGE_COUNT + 1)
        }

        for future in as_completed(futures):
            page = futures[future]
            try:
                response = future.result()
                rows = records_from_response(response)

                if not rows:
                    raise ValueError("No order records found in response")

                pages[page] = rows
            except Exception as exc:
                failures.append((page, str(exc)))

    if failures:
        raise RuntimeError(f"Failed to load order pages: {failures[:10]}")

    combined = []
    for page in range(1, PAGE_COUNT + 1):
        combined.extend(pages[page])

    if not combined:
        raise RuntimeError("No orders returned by the API")

    df = pd.DataFrame(combined)

    id_column = next(
        (column for column in ("id", "order_id", "orderId") if column in df.columns),
        None,
    )

    if id_column is None:
        raise RuntimeError(f"No order ID column found. Columns: {list(df.columns)}")

    if "updated_at" in df.columns:
        df["_updated_sort"] = pd.to_datetime(
            df["updated_at"], errors="coerce", utc=True
        )
        df = df.sort_values("_updated_sort", na_position="first")

    df = df.drop_duplicates(subset=[id_column], keep="last")
    df = df.drop(columns=["_updated_sort"], errors="ignore")

    if "created_at" in df.columns:
        df["_date"] = pd.to_datetime(
            df["created_at"], errors="coerce", utc=True
        ).dt.tz_convert("Asia/Kolkata").dt.date

    if "amount" in df.columns:
        df["_amount"] = pd.to_numeric(df["amount"], errors="coerce")
    elif {"qty", "unit_price"}.issubset(df.columns):
        df["_amount"] = (
            pd.to_numeric(df["qty"], errors="coerce")
            * pd.to_numeric(df["unit_price"], errors="coerce")
        )
    else:
        raise RuntimeError("Orders contain neither amount nor qty/unit_price")

    if "status" in df.columns:
        df["_status"] = df["status"].astype(str).str.casefold()
    else:
        df["_status"] = ""

    if "currency" in df.columns:
        df["_currency"] = df["currency"].astype(str).str.upper()
    else:
        df["_currency"] = "USD"

    return df, len(pages)


def load_snapshot():
    """Build a complete new snapshot without disturbing the current one."""
    validate_context_url()
    root = get_api("/")
    orders, pages_loaded = load_orders()

    rates = discover_rates(root)

    # Try documented rate resources if the root has no embedded rates.
    if not rates:
        candidate_paths = sorted(
            path for path in find_resource_paths(root)
            if "rate" in path.casefold() or "currency" in path.casefold()
        )

        for path in candidate_paths:
            try:
                rates = discover_rates(get_api(path))
                if rates:
                    break
            except Exception:
                logging.exception("Could not load rate resource %s", path)

    if not rates:
        raise RuntimeError(
            "No usd_per_unit exchange-rate mapping was found in the API root "
            "or documented rate resources. Inspect the API response before proceeding."
        )

    rates.setdefault("USD", 1.0)

    # Validate currencies before publishing the snapshot.
    missing = sorted(set(orders["_currency"].dropna()) - set(rates))
    if missing:
        raise RuntimeError(f"Missing exchange rates for currencies: {missing}")

    orders["_amount_usd"] = (
        orders["_amount"] * orders["_currency"].map(rates)
    )

    # Build a candidate first; replace the active snapshot only on success.
    return {
        "orders": orders,
        "rates": rates,
        "root": root,
        "loaded_at": datetime.now(TZ).isoformat(),
        "order_count": len(orders),
        "page_count": pages_loaded,
        "error": None,
    }


def ensure_loaded():
    with data_lock:
        if snapshot["loaded_at"] is not None:
            return

    with refresh_lock:
        with data_lock:
            if snapshot["loaded_at"] is not None:
                return

        new_snapshot = load_snapshot()

        with data_lock:
            snapshot.update(new_snapshot)

    logging.info(
        "Ledger ready: %s unique orders, %s pages",
        snapshot["order_count"],
        snapshot["page_count"],
    )


def refresh_snapshot():
    with refresh_lock:
        new_snapshot = load_snapshot()

        with data_lock:
            snapshot.update(new_snapshot)

    return new_snapshot


def interpret(question):
    result = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0,
        max_tokens=350,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": QUERY_PROMPT},
            {"role": "user", "content": question},
        ],
        timeout=10,
    )

    plan = json.loads(result.choices[0].message.content)

    allowed_metrics = {
        "revenue", "refund_count", "refund_amount", "order_count",
        "top_product", "top_customer", "top_region",
        "product_revenue", "customer_spending", "unknown",
    }

    if plan.get("metric") not in allowed_metrics:
        plan["metric"] = "unknown"

    plan["limit"] = max(1, min(int(plan.get("limit") or 1), 20))
    return plan


def apply_filters(df, plan):
    filtered = df.copy()

    for field in ("region", "product", "customer"):
        value = plan.get(field)
        if value:
            if field not in filtered.columns:
                return filtered.iloc[0:0]

            filtered = filtered[
                filtered[field].astype(str).str.casefold()
                == str(value).casefold()
            ]

    start = plan.get("start_date")
    end = plan.get("end_date")

    if start or end:
        if "_date" not in filtered.columns:
            return filtered.iloc[0:0]

        if start:
            filtered = filtered[filtered["_date"] >= date.fromisoformat(start)]

        if end:
            filtered = filtered[filtered["_date"] <= date.fromisoformat(end)]

    status = plan.get("status")
    if status:
        filtered = filtered[
            filtered["_status"] == str(status).casefold()
        ]

    return filtered


def calculate(plan, orders):
    metric = plan["metric"]
    df = apply_filters(orders, plan)

    paid = df[df["_status"] == "paid"]
    refunded = df[df["_status"] == "refunded"]

    if metric == "revenue":
        return round(float(paid["_amount_usd"].sum()), 2)

    if metric == "refund_count":
        return int(len(refunded))

    if metric == "refund_amount":
        return round(float(refunded["_amount_usd"].sum()), 2)

    if metric == "order_count":
        return int(len(df))

    ranking = {
        "top_product": "product",
        "top_customer": "customer",
        "top_region": "region",
    }

    if metric in ranking:
        column = ranking[metric]

        if column not in paid.columns:
            return None

        totals = (
            paid.groupby(column, dropna=True)["_amount_usd"]
            .sum()
            .sort_values(ascending=False)
        )

        if totals.empty:
            return None

        if plan["limit"] == 1:
            return str(totals.index[0])

        return [
            {"name": str(name), "revenue_usd": round(float(value), 2)}
            for name, value in totals.head(plan["limit"]).items()
        ]

    if metric in ("product_revenue", "customer_spending"):
        return round(float(paid["_amount_usd"].sum()), 2)

    return None


@app.get("/")
def home():
    return jsonify({
        "service": "acme-ledger-api",
        "usage": "POST JSON {\"question\":\"...\"} to this URL",
    })


@app.get("/health")
def health():
    loaded = snapshot["loaded_at"] is not None
    return jsonify({
        "status": "ready" if loaded else "not_ready",
        "loaded_at": snapshot["loaded_at"],
        "order_count": snapshot["order_count"],
        "pages_loaded": snapshot["page_count"],
        "error": snapshot["error"],
    }), 200 if loaded else 503


@app.post("/refresh")
def refresh():
    if not REFRESH_TOKEN:
        return jsonify({"error": "Refresh endpoint is disabled"}), 403

    supplied = request.headers.get("X-Refresh-Token", "")
    if supplied != REFRESH_TOKEN:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        result = refresh_snapshot()
        return jsonify({
            "status": "refreshed",
            "loaded_at": result["loaded_at"],
            "order_count": result["order_count"],
            "pages_loaded": result["page_count"],
        })
    except Exception as exc:
        logging.exception("Refresh failed")
        return jsonify({"error": str(exc)}), 500


@app.post("/")
def answer():
    body = request.get_json(silent=True) or {}
    question = body.get("question")

    if not isinstance(question, str) or not question.strip():
        return jsonify({"error": "question must be a non-empty string"}), 400

    try:
        ensure_loaded()

        with data_lock:
            orders = snapshot["orders"].copy()

        plan = interpret(question)
        result = calculate(plan, orders)

        if result is None:
            return jsonify({
                "answer": "The question could not be answered from the available ledger schema."
            })

        return jsonify({"answer": result})

    except Exception as exc:
        logging.exception("Question failed")
        return jsonify({
            "error": "Unable to answer the question",
            "detail": str(exc),
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "10000")),
        debug=False,
    )
