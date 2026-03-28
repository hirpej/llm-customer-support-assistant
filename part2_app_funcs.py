import os
import json
import re
from datetime import datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI


# ----------------------------
# Helpers
# ----------------------------

def render_output(output: dict, debug_raw: str | None = None):
    st.subheader("Result")
    st.write("Final answer:")
    st.write(output.get("final_answer", ""))

    # Optional debug raw response
    if debug_raw:
        with st.expander("DEBUG raw response"):
            st.code(debug_raw, language="json")

    # Tool calls table
    st.subheader("Tool calls")
    tool_calls = output.get("tool_calls", [])
    if tool_calls:
        df = pd.DataFrame(tool_calls)
        st.dataframe(df, use_container_width=True)
    else:
        st.caption("No tool calls yet.")

    # Output JSON
    st.subheader("Output JSON")
    st.code(json.dumps(output, indent=2, ensure_ascii=False), language="json")




def mask_email(email: str) -> str:
    if not isinstance(email, str) or "@" not in email:
        return ""
    local, domain = email.split("@", 1)
    if len(local) <= 1:
        masked_local = "*"
    elif len(local) == 2:
        masked_local = local[0] + "*"
    else:
        masked_local = local[0] + ("*" * (len(local) - 2)) + local[-1]
    return f"{masked_local}@{domain}"


def normalize_order_id(x: str) -> str:
    if not isinstance(x, str):
        x = str(x)
    return re.sub(r"\s+", "", x).upper()


def normalize_customer_id(x: str) -> str:
    # keep it simple; many CSVs store IDs as strings/ints inconsistently
    if x is None:
        return ""
    x = str(x).strip()
    return x


# ----------------------------
# Load .env + Client
# ----------------------------
load_dotenv()

ARVAN_API_KEY = os.getenv("ARVAN_API_KEY", "").strip()
ARVAN_BASE_URL = os.getenv("ARVAN_BASE_URL", "").strip()
ARVAN_MODEL = os.getenv("ARVAN_MODEL", "").strip()

LLM_READY = bool(ARVAN_API_KEY and ARVAN_BASE_URL and ARVAN_MODEL)

client = None
if LLM_READY:
    client = OpenAI(api_key=ARVAN_API_KEY, base_url=ARVAN_BASE_URL)


# ----------------------------
# Load data (from ./data)
# ----------------------------
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")

customers_path = os.path.join(DATA_DIR, "customers.csv")
orders_path = os.path.join(DATA_DIR, "orders.csv")

data_ok = True
customers_df = None
orders_df = None

try:
    customers_df = pd.read_csv(customers_path)
    orders_df = pd.read_csv(orders_path)
except Exception as e:
    data_ok = False
    st.error(f"Failed to load data files: {e}")

# Normalize column names to lowercase for robustness
if data_ok:
    customers_df.columns = [c.strip().lower() for c in customers_df.columns]
    orders_df.columns = [c.strip().lower() for c in orders_df.columns]

    # Try to standardize expected column names
    # customers: customer_id, email (sometimes "id" or "customerid")
    if "customer_id" not in customers_df.columns:
        for alt in ["id", "customerid", "customer_id "]:
            if alt in customers_df.columns:
                customers_df = customers_df.rename(columns={alt: "customer_id"})
                break

    if "email" not in customers_df.columns:
        for alt in ["mail", "e-mail"]:
            if alt in customers_df.columns:
                customers_df = customers_df.rename(columns={alt: "email"})
                break

    # orders: order_id, customer_id, status, total, date
    if "order_id" not in orders_df.columns:
        for alt in ["id", "orderid", "order_id "]:
            if alt in orders_df.columns:
                orders_df = orders_df.rename(columns={alt: "order_id"})
                break

    if "customer_id" not in orders_df.columns:
        for alt in ["customerid", "customer", "cust_id"]:
            if alt in orders_df.columns:
                orders_df = orders_df.rename(columns={alt: "customer_id"})
                break

    if "total" not in orders_df.columns:
        for alt in ["amount", "price", "sum"]:
            if alt in orders_df.columns:
                orders_df = orders_df.rename(columns={alt: "total"})
                break

    # Create normalized keys for matching
    customers_df["__customer_id_norm__"] = customers_df["customer_id"].apply(normalize_customer_id)
    orders_df["__order_id_norm__"] = orders_df["order_id"].apply(normalize_order_id)
    orders_df["__customer_id_norm__"] = orders_df["customer_id"].apply(normalize_customer_id)


# ----------------------------
# Tools implementation
# ----------------------------
def tool_get_order(order_id: str, customers_df: pd.DataFrame, orders_df: pd.DataFrame) -> dict:
    oid = normalize_order_id(order_id)

    hit = orders_df[orders_df["__order_id_norm__"] == oid]
    if hit.empty:
        return {"found": False, "order_id": oid}

    row = hit.iloc[0]
    cid = row.get("__customer_id_norm__", "")

    email = ""
    if cid and ("__customer_id_norm__" in customers_df.columns):
        c_hit = customers_df[customers_df["__customer_id_norm__"] == cid]
        if not c_hit.empty and "email" in c_hit.columns:
            email = str(c_hit.iloc[0].get("email", "")).strip()

    masked = mask_email(email)

    return {
        "found": True,
        "order_id": str(row.get("order_id", oid)),
        "status": str(row.get("status", "")).strip(),
        "total": float(row.get("total", 0) or 0),
        "masked_email": masked,
    }


def tool_refund_order(order_id: str, amount: float, customers_df: pd.DataFrame, orders_df: pd.DataFrame) -> dict:
    oid = normalize_order_id(order_id)

    hit = orders_df[orders_df["__order_id_norm__"] == oid]
    if hit.empty:
        return {"ok": False, "reason": "order_not_found", "order_id": oid}

    row = hit.iloc[0]
    status = str(row.get("status", "")).strip().lower()
    total = float(row.get("total", 0) or 0)

    if status not in ["settled", "prepping"]:
        return {"ok": False, "reason": f"refund_not_allowed_for_status:{status}", "order_id": oid}

    if amount <= 0:
        return {"ok": False, "reason": "amount_must_be_positive", "order_id": oid}

    if amount > total:
        return {"ok": False, "reason": "amount_exceeds_total", "order_id": oid, "total": total}

    # Mock refund success (no database write)
    return {"ok": True, "order_id": oid, "refunded": float(amount)}


def tool_spend_in_period(customer_id: str, start: str, end: str, customers_df: pd.DataFrame, orders_df: pd.DataFrame) -> dict:
    cid = normalize_customer_id(customer_id)

    try:
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")
    except Exception:
        return {"ok": False, "reason": "invalid_date_format_use_YYYY-MM-DD"}

    if "date" not in orders_df.columns:
        return {"ok": False, "reason": "orders_csv_missing_date_column"}

    tmp = orders_df.copy()
    tmp["__date_dt__"] = pd.to_datetime(tmp["date"], errors="coerce")

    filt = (tmp["__customer_id_norm__"] == cid) & (tmp["__date_dt__"] >= start_dt) & (tmp["__date_dt__"] <= end_dt)
    spend = float(tmp.loc[filt, "total"].fillna(0).sum())

    return {"ok": True, "customer_id": cid, "start": start, "end": end, "spend": spend}


def call_tool(name: str, args: dict, customers_df: pd.DataFrame, orders_df: pd.DataFrame) -> dict:
    if name == "get_order":
        return tool_get_order(args.get("order_id", ""), customers_df, orders_df)
    if name == "refund_order":
        return tool_refund_order(args.get("order_id", ""), float(args.get("amount", 0)), customers_df, orders_df)
    if name == "spend_in_period":
        return tool_spend_in_period(args.get("customer_id", ""), args.get("start", ""), args.get("end", ""), customers_df, orders_df)
    return {"ok": False, "reason": "unknown_tool"}


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "Get an order by order_id. Returns status, total, and masked_email.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "refund_order",
            "description": "Refund credits for an order. Allowed only for status settled/prepping and amount <= total.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "amount": {"type": "number"},
                },
                "required": ["order_id", "amount"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spend_in_period",
            "description": "Compute total spend for a customer in [start,end] (YYYY-MM-DD).",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string"},
                    "start": {"type": "string"},
                    "end": {"type": "string"},
                },
                "required": ["customer_id", "start", "end"],
            },
        },
    },
]


# ---------- UI 
st.set_page_config(page_title="Starship Coffee — Customer & Orders Assistant", layout="wide")
st.title("Starship Coffee — Customer & Orders Assistant")

st.sidebar.header("Mode")
use_llm = st.sidebar.toggle("Use LLM tool calling (needs billing)", value=True)

if not data_ok:
    st.stop()

# Session state init (safe)
if "question" not in st.session_state:
    st.session_state["question"] = ""

def set_q(text: str) -> None:
    st.session_state["question"] = text
    st.rerun()

st.subheader("Ask a Question")
st.text_area("Your question", key="question", height=90)

st.subheader("Presets")
c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
with c1:
    st.button(
        "Preset 1 - Spend in period",
        on_click=set_q,
        args=("How much did customer 1 spend between 2025-10-01 and 2025-10-31?",),
    )
with c2:
    st.button(
        "Preset 2 - Refund",
        on_click=set_q,
        args=("Refund 5.40 credits for order B77.",),
    )
with c3:
    st.button(
        "Preset 3 - Order status",
        on_click=set_q,
        args=("What is the status and masked email for order c9?",),
    )
with c4:
    run = st.button("Run", type="primary")

st.divider()

# ---------- Helpers ----------

def _dump_debug(raw_debug) -> str:
    """Return JSON string for debug panel."""
    if raw_debug is None:
        return "None"
    if hasattr(raw_debug, "model_dump_json"):
        return raw_debug.model_dump_json(indent=2)
    try:
        return json.dumps(raw_debug, indent=2, ensure_ascii=False)
    except Exception:
        return str(raw_debug)

def run_mock(q: str):
    q_low = (q or "").lower().strip()
    tool_calls_log = []

    # Order status: "status ... order c9"
    if ("status" in q_low) and ("order" in q_low):
        m = re.search(r"order\s+([a-z]\d+)", q_low)
        if m:
            oid = m.group(1).strip().upper()
            result = call_tool("get_order", {"order_id": oid}, customers_df, orders_df)
            tool_calls_log.append({"tool": "get_order", "args": {"order_id": oid}, "result": result})

            if result.get("found"):
                final_text = f"Order {result.get('order_id', oid)} is {result.get('status','')}. Contact: {result.get('masked_email','')}."
            else:
                final_text = f"Order {oid} not found."
            return final_text, tool_calls_log

    # Refund: "Refund 5.40 credits for order B77."
    if q_low.startswith("refund"):
        m = re.search(r"refund\s+([\d\.]+)\s+credits\s+for\s+order\s+([a-z]\d+)", q_low)
        if m:
            amount = float(m.group(1))
            oid = m.group(2).strip().upper()
            result = call_tool("refund_order", {"order_id": oid, "amount": amount}, customers_df, orders_df)
            tool_calls_log.append({"tool": "refund_order", "args": {"order_id": oid, "amount": amount}, "result": result})

            if result.get("ok"):
                return "{ok:true}", tool_calls_log
            return "{ok:false}", tool_calls_log

    return "Could not understand your request (mock mode).", tool_calls_log


def run_llm(q: str):
    system_msg = (
        "You are a helpful assistant for Starship Coffee.\n"
        "Use tool calling when needed.\n"
        "Rules:\n"
        "- Only use the provided tools.\n"
        "- Always return masked emails.\n"
        "- Keep answers short.\n"
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": q},
    ]

    tool_calls_log = []
    raw_debug = None

    # If LLM not ready -> fallback
    if (not LLM_READY) or (client is None):
        final_text, tool_calls_log = run_mock(q)
        return final_text, tool_calls_log, {"note": "LLM unavailable; used mock."}

    # Tool loop (max 3 rounds)
    for _ in range(3):
        resp = client.chat.completions.create(
            model=ARVAN_MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        raw_debug = resp

        msg = resp.choices[0].message

        # Tool calls
        if msg.tool_calls:
            messages.append(msg)

            for tc in msg.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments)

                result = call_tool(name, args, customers_df, orders_df)
                tool_calls_log.append({"tool": name, "args": args, "result": result})

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result),
                    }
                )
            continue

        # Final text
        final_text = (msg.content or "").strip()
        return final_text, tool_calls_log, raw_debug

    return "No final answer returned (model kept requesting tools).", tool_calls_log, raw_debug


# ---------- Run ----------
if run:
    q = (st.session_state.get("question") or "").strip()
    if not q:
        st.warning("Please type a question or click a preset.")
        st.stop()

    if use_llm:
        final_text, tool_calls_log, raw_debug = run_llm(q)
    else:
        final_text, tool_calls_log = run_mock(q)
        raw_debug = {"note": "mock mode"}

    # Render
    st.subheader("Result")
    st.write("Final answer:")
    st.write(final_text if final_text else "(empty)")

    with st.expander("DEBUG raw response"):
        st.code(_dump_debug(raw_debug), language="json")

    st.subheader("Tool calls")
    if tool_calls_log:
        st.dataframe(pd.DataFrame(tool_calls_log))
    else:
        st.caption("No tool calls yet.")

    st.subheader("Output JSON")
    st.code(
        json.dumps({"final_answer": final_text, "tool_calls": tool_calls_log}, indent=2),
        language="json",
    )
