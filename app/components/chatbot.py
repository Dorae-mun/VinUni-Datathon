import os
import html
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
import streamlit as st

from src.data_loader import load_sales_summary
from src.rag import RagRetriever, build_augmented_user_message

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DOCS_DIR = PROJECT_ROOT / "data" / "raw_docs"
RAG_DIR = PROJECT_ROOT / "data" / "rag"
DEFAULT_MODEL = "openai/gpt-4o-mini"
FALLBACK_MODELS = ["openai/gpt-4.1-mini", "openai/gpt-4o"]
DEFAULT_TEMPERATURE = 0.2
DEFAULT_USE_RAG = True
DEFAULT_TOP_K = 4
MAX_CHAT_TURNS = 12


def _format_currency(v: float) -> str:
    return f"{v:,.0f} VND"


def _get_openrouter_api_key() -> Optional[str]:
    # Prefer Streamlit secrets, fallback to env var.
    try:
        key = st.secrets.get("OPENROUTER_API_KEY")  # type: ignore[attr-defined]
        if key:
            return str(key)
    except Exception:
        pass
    return os.getenv("OPENROUTER_API_KEY")


def _build_data_context(data: Dict[str, pd.DataFrame]) -> str:
    lines: List[str] = []
    lines.append("You are an analytics copilot for a retail dashboard.")
    lines.append("Answer in Vietnamese. Be concise, actionable, and data-driven.")
    lines.append("")
    lines.append("Available tables (name: rows x cols, sample columns):")
    for name, df in data.items():
        if df is None:
            continue
        cols = list(df.columns)
        sample_cols = ", ".join(cols[:12]) + ("..." if len(cols) > 12 else "")
        lines.append(f"- {name}: {df.shape[0]} x {df.shape[1]} | {sample_cols}")

    sales = data.get("sales")
    if sales is not None and not sales.empty and "Date" in sales.columns:
        summary = load_sales_summary(sales)
        lines.append("")
        lines.append("Sales quick summary:")
        lines.append(f"- Date range: {summary['date_start'].date()} → {summary['date_end'].date()}")
        lines.append(f"- Total revenue: {summary['total_revenue']:,.0f}")
        lines.append(f"- Gross profit: {summary['gross_profit']:,.0f}")
        lines.append(f"- Gross margin %: {summary['gross_margin_pct']:.2f}")

        monthly = (
            sales.assign(year_month=sales["Date"].dt.to_period("M").astype(str))
            .groupby("year_month", as_index=False)["Revenue"]
            .sum()
            .sort_values("year_month")
        )
        if len(monthly) >= 6:
            first3 = monthly["Revenue"].head(3).mean()
            last3 = monthly["Revenue"].tail(3).mean()
            trend = "tăng" if last3 > first3 else "giảm"
            pct = ((last3 - first3) / first3 * 100) if first3 else 0.0
            lines.append(f"- Monthly revenue trend (3M vs first 3M): {trend} {pct:.2f}%")

    products = data.get("products")
    order_items = data.get("order_items")
    if (
        products is not None
        and order_items is not None
        and not products.empty
        and not order_items.empty
        and "line_revenue" in order_items.columns
        and "product_id" in order_items.columns
        and "product_id" in products.columns
        and "product_name" in products.columns
    ):
        product_revenue = (
            order_items.merge(
                products[["product_id", "product_name"]],
                on="product_id",
                how="left",
            )
            .groupby("product_name", as_index=False)["line_revenue"]
            .sum()
            .sort_values("line_revenue", ascending=False)
        )
        lines.append("")
        lines.append("Top products by revenue:")
        for _, row in product_revenue.head(8).iterrows():
            lines.append(f"- {row['product_name']}: {row['line_revenue']:,.0f}")

    lines.append("")
    lines.append("Use retrieved document context when available and cite source file names in your answer.")
    lines.append("")
    lines.append("Prefer answering directly with numbers if data context already contains the needed metrics.")
    return "\n".join(lines)


def _try_answer_structured_query(query: str, data: Dict[str, pd.DataFrame]) -> Optional[str]:
    q = query.lower().strip()
    sales = data.get("sales")

    if sales is not None and not sales.empty and "revenue" in q and "tổng" in q and "mặt hàng" not in q:
        total_rev = float(sales["Revenue"].sum())
        return f"Tổng doanh thu của doanh nghiệp là **{_format_currency(total_rev)}**."

    if sales is not None and not sales.empty and ("xu hướng" in q or "trend" in q) and "doanh thu" in q:
        monthly = (
            sales.assign(year_month=sales["Date"].dt.to_period("M").astype(str))
            .groupby("year_month", as_index=False)["Revenue"]
            .sum()
            .sort_values("year_month")
        )
        if len(monthly) >= 6:
            first3 = float(monthly["Revenue"].head(3).mean())
            last3 = float(monthly["Revenue"].tail(3).mean())
            trend = "tăng" if last3 > first3 else "giảm"
            pct = ((last3 - first3) / first3 * 100) if first3 else 0.0
            return (
                f"Xu hướng doanh thu tổng thể đang **{trend}**.\n\n"
                f"- TB 3 tháng đầu: **{_format_currency(first3)}**\n"
                f"- TB 3 tháng gần nhất: **{_format_currency(last3)}**\n"
                f"- Mức thay đổi: **{pct:.2f}%**"
            )

    if ("mặt hàng" in q or "sản phẩm" in q) and ("doanh thu" in q):
        products = data.get("products")
        order_items = data.get("order_items")
        if (
            products is not None
            and order_items is not None
            and not products.empty
            and not order_items.empty
            and "line_revenue" in order_items.columns
        ):
            product_revenue = (
                order_items.merge(
                    products[["product_id", "product_name"]],
                    on="product_id",
                    how="left",
                )
                .groupby("product_name", as_index=False)["line_revenue"]
                .sum()
                .sort_values("line_revenue", ascending=False)
            )
            top = product_revenue.head(10)
            lines = ["Top 10 mặt hàng theo doanh thu:"]
            for _, row in top.iterrows():
                lines.append(f"- {row['product_name']}: {_format_currency(float(row['line_revenue']))}")
            if not top.empty:
                lines.append(f"\nMặt hàng thế mạnh hiện tại: **{top.iloc[0]['product_name']}**.")
            return "\n".join(lines)

    if ("lợi nhuận" in q or "profit" in q) and ("địa lý" in q or "khu vực" in q or "region" in q or "vị trí" in q):
        orders = data.get("orders")
        order_items = data.get("order_items")
        products = data.get("products")
        geography = data.get("geography")
        if (
            orders is not None
            and order_items is not None
            and products is not None
            and geography is not None
            and not orders.empty
            and not order_items.empty
            and not products.empty
            and not geography.empty
        ):
            merged = (
                order_items.merge(
                    products[["product_id", "cogs"]],
                    on="product_id",
                    how="left",
                )
                .merge(
                    orders[["order_id", "zip"]],
                    on="order_id",
                    how="left",
                )
                .merge(
                    geography[["zip", "region"]],
                    on="zip",
                    how="left",
                )
            )
            merged["line_cogs"] = merged["quantity"] * merged["cogs"]
            merged["line_profit"] = merged["line_revenue"] - merged["line_cogs"]
            by_region = (
                merged.groupby("region", as_index=False)["line_profit"]
                .sum()
                .sort_values("line_profit", ascending=False)
            )
            if not by_region.empty:
                lines = ["Lợi nhuận theo khu vực địa lý (region):"]
                for _, row in by_region.head(10).iterrows():
                    lines.append(f"- {row['region']}: {_format_currency(float(row['line_profit']))}")
                lines.append(f"\nKhu vực lợi nhuận cao nhất: **{by_region.iloc[0]['region']}**.")
                return "\n".join(lines)

    return None


@st.cache_resource(show_spinner=False)
def _load_retriever() -> RagRetriever:
    retriever = RagRetriever(RAG_DIR)
    retriever.load()
    return retriever


def _openrouter_chat(
    api_key: str,
    messages: List[dict],
    model: str,
    temperature: float = 0.2,
) -> str:
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def _build_request_messages(chat_messages: List[dict]) -> List[dict]:
    # Keep system prompt + latest conversation turns to reduce context overflow risk.
    if not chat_messages:
        return chat_messages
    system = [m for m in chat_messages if m.get("role") == "system"][:1]
    non_system = [m for m in chat_messages if m.get("role") != "system"]
    recent = non_system[-MAX_CHAT_TURNS:]
    return system + recent


def _call_openrouter_with_fallback(
    api_key: str,
    messages: List[dict],
    primary_model: str,
    temperature: float,
) -> str:
    model_candidates = [primary_model] + [m for m in FALLBACK_MODELS if m != primary_model]
    last_error = "Không rõ nguyên nhân"
    for model_name in model_candidates:
        try:
            return _openrouter_chat(
                api_key=api_key,
                messages=messages,
                model=model_name,
                temperature=temperature,
            )
        except requests.HTTPError as e:
            body = ""
            if e.response is not None:
                try:
                    body = e.response.text
                except Exception:
                    body = ""
            last_error = f"HTTP {e.response.status_code if e.response is not None else 'unknown'} - {body[:280]}"
            continue
        except Exception as e:
            last_error = str(e)
            continue
    return f"Lỗi gọi OpenRouter sau khi thử nhiều model: {last_error}"


def render_chatbot(data: Dict[str, pd.DataFrame]):
    RAW_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    RAG_DIR.mkdir(parents=True, exist_ok=True)

    api_key = _get_openrouter_api_key()
    if not api_key:
        st.sidebar.info(
            "Chưa cấu hình OpenRouter API key. "
            "Hãy set environment variable `OPENROUTER_API_KEY` "
            "hoặc thêm vào `.streamlit/secrets.toml`."
        )
        return

    if "chat_messages" not in st.session_state:
        system_context = _build_data_context(data)
        st.session_state.chat_messages = [
            {"role": "system", "content": system_context},
            {"role": "assistant", "content": "Bạn muốn hỏi gì về dữ liệu kinh doanh?"},
        ]
    if "chat_input_nonce" not in st.session_state:
        st.session_state.chat_input_nonce = 0

    retriever = _load_retriever()
    rag_ready = retriever.is_ready()

    input_key = f"chat_user_text_{st.session_state.chat_input_nonce}"
    user_text = st.sidebar.text_area(
        "Nhập câu hỏi",
        key=input_key,
        height=80,
        placeholder="Ví dụ: Vì sao doanh thu giảm tuần gần đây?",
    )
    col_a, col_b = st.sidebar.columns([1, 1])
    send = col_a.button("Send", use_container_width=True)
    clear = col_b.button("Clear", use_container_width=True)

    if clear:
        system_context = _build_data_context(data)
        st.session_state.chat_messages = [
            {"role": "system", "content": system_context},
            {"role": "assistant", "content": "Bạn muốn hỏi gì về dữ liệu kinh doanh?"},
        ]
        st.session_state.chat_input_nonce += 1
        st.rerun()

    if send and user_text.strip():
        clean_user_text = user_text.strip()
        st.session_state.chat_messages.append({"role": "user", "content": clean_user_text})
        direct_answer = _try_answer_structured_query(clean_user_text, data)
        if direct_answer is not None:
            answer = direct_answer
        else:
            request_messages = _build_request_messages(st.session_state.chat_messages)

            if DEFAULT_USE_RAG and rag_ready:
                try:
                    chunks = retriever.retrieve(clean_user_text, top_k=DEFAULT_TOP_K)
                    request_messages[-1] = {
                        "role": "user",
                        "content": build_augmented_user_message(clean_user_text, chunks),
                    }
                except Exception:
                    # Fallback to plain user question if retrieval fails.
                    pass

            with st.spinner("Thinking..."):
                answer = _call_openrouter_with_fallback(
                    api_key=api_key,
                    messages=request_messages,
                    primary_model=DEFAULT_MODEL,
                    temperature=DEFAULT_TEMPERATURE,
                )

        st.session_state.chat_messages.append({"role": "assistant", "content": answer})
        st.session_state.chat_input_nonce += 1

    history_box = st.sidebar.container(height=410, border=False)
    with history_box:
        st.markdown(
            """
            <style>
            .chat-wrap {display:flex; margin:8px 0;}
            .chat-wrap.user {justify-content:flex-end;}
            .chat-wrap.assistant {justify-content:flex-start;}
            .chat-bubble {
                max-width: 92%;
                padding: 10px 12px;
                border-radius: 14px;
                font-size: 0.92rem;
                line-height: 1.4;
                word-break: break-word;
                white-space: pre-wrap;
            }
            .chat-bubble.user {
                background: #dbeafe;
                border: 1px solid #93c5fd;
                color: #0f172a;
                border-bottom-right-radius: 6px;
            }
            .chat-bubble.assistant {
                background: #f3f4f6;
                border: 1px solid #e5e7eb;
                color: #111827;
                border-bottom-left-radius: 6px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        for m in st.session_state.chat_messages:
            if m["role"] == "system":
                continue
            role = "user" if m["role"] == "user" else "assistant"
            st.markdown(
                f"""
                <div class="chat-wrap {role}">
                    <div class="chat-bubble {role}">{html.escape(str(m["content"]))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
