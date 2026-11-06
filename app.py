"""
PhoBot - Chatbot nhà hàng Viet Cuisine (Streamlit + Gemini)
Thực đơn được import trong code từ MENU_PATH, không có phần nhập/tải thực đơn trên giao diện.
Chạy:  streamlit run app.py
"""
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv(override=True)

# ----------------------------- CẤU HÌNH ----------------------------------
DEFAULT_MODEL = "gemini-3.6-flash"
BOT_NAME = "PhoBot"
RESTAURANT = "Viet Cuisine"

# Đường dẫn file thực đơn (cột: name, description, ingredients, notes).
# Sửa đường dẫn này trong code nếu cần.
MENU_PATH = Path('./menu.csv').parent / "menu.csv"

SYSTEM_TEMPLATE = """Bạn là {bot}, trợ lý AI thân thiện của nhà hàng {restaurant}.
Chỉ trả lời dựa trên thực đơn bên dưới. Nếu thông tin không có trong thực đơn,
hãy nói rõ là bạn chưa có thông tin và gợi ý khách hỏi nhân viên.
Luôn trả lời bằng tiếng Việt, lịch sự, ngắn gọn, dùng markdown khi cần.

=== THỰC ĐƠN ===
{menu}
"""

SUGGESTIONS = [
    "🍜 Hãy nói về món bún bò",
    "🥗 Có món nào phù hợp cho người ăn nhẹ?",
    "🍚 Cơm tấm gồm những gì?",
    "⭐ Bạn gợi ý món nào nổi bật nhất?",
]

st.set_page_config(page_title=f"{BOT_NAME} | {RESTAURANT}", page_icon="🍜", layout="centered")

# ------------------------------- CSS --------------------------------------
st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; max-width: 820px;}
      .hero {
          background: linear-gradient(135deg, #e8590c 0%, #f59f00 100%);
          color: white; padding: 1.4rem 1.6rem; border-radius: 18px;
          margin-bottom: 1.2rem; box-shadow: 0 6px 18px rgba(232,89,12,.25);
      }
      .hero h1 {margin: 0; font-size: 1.8rem; color: white;}
      .hero p  {margin: .3rem 0 0; opacity: .92;}
      div[data-testid="stChatMessage"] {border-radius: 14px; padding: .6rem .8rem;}
      .stButton > button {border-radius: 999px; text-align: left;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------- XỬ LÝ THỰC ĐƠN -------------------------------
def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def add_detail(item_text: str, value, label: str) -> str:
    value = clean_text(value)
    if not value:
        return item_text
    suffix = "" if value.endswith((".", "!", "?")) else "."
    return f"{item_text} {label}; {value}{suffix}"


def build_menu_context(menu_df: pd.DataFrame) -> str:
    """Chuyển dữ liệu menu thành đoạn ngữ cảnh cho chatbot."""
    lines = []
    for _, row in menu_df.iterrows():
        item = f"Tên: {clean_text(row.get('name'))}: {clean_text(row.get('description'))}"
        item = add_detail(item, row.get("ingredients"), "Thành phần")
        item = add_detail(item, row.get("notes"), "Ghi chú")
        lines.append(item)
    return "\n".join(lines)


@st.cache_data(show_spinner=False)
def load_menu_context(path: str) -> str:
    return build_menu_context(pd.read_csv(path, index_col=0))


if not MENU_PATH.exists():
    st.error(f"Không tìm thấy file thực đơn: `{MENU_PATH}`. Hãy đặt `menu.csv` cạnh `app.py` hoặc sửa `MENU_PATH` trong code.")
    st.stop()

menu_context = load_menu_context(str(MENU_PATH))


# --------------------------- GỌI GEMINI -----------------------------------
@st.cache_resource(show_spinner=False)
def get_client(api_key: str):
    return genai.Client(api_key=api_key)


MOCK_REPLY = (
    f"Xin chào, mình là {BOT_NAME}. Đây là phản hồi mẫu trong chế độ Mock. "
    "Khi có API key hợp lệ, câu trả lời sẽ được thay bằng phản hồi từ Gemini."
)


def stream_answer(history, model, api_key, use_mock):
    """Sinh câu trả lời theo dạng stream, có nhớ lịch sử hội thoại."""
    if use_mock or not api_key:
        yield MOCK_REPLY
        return

    system_instruction = SYSTEM_TEMPLATE.format(
        bot=BOT_NAME, restaurant=RESTAURANT, menu=menu_context
    )
    contents = [
        types.Content(
            role="user" if m["role"] == "user" else "model",
            parts=[types.Part(text=m["content"])],
        )
        for m in history
    ]
    try:
        client = get_client(api_key)
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(system_instruction=system_instruction),
        ):
            if chunk.text:
                yield chunk.text
    except Exception as err:  # noqa: BLE001
        yield f"⚠️ Không gọi được Gemini: `{err}`\n\n{MOCK_REPLY}"


# ------------------------------ SIDEBAR -----------------------------------
with st.sidebar:
    st.header("⚙️ Cài đặt")

    env_key = os.getenv("GEMINI_API_KEY", "")
    api_key = st.text_input(
        "Gemini API Key", value=env_key, type="password",
        help="Mặc định đọc từ biến môi trường GEMINI_API_KEY (file .env).",
    )
    model_name = st.text_input("Model", value=DEFAULT_MODEL)
    use_mock = st.toggle("Chế độ Mock (không gọi API)", value=not bool(api_key))

    st.divider()
    if st.button("🗑️ Xóa cuộc trò chuyện", width="stretch"):
        st.session_state.pop("messages", None)
        st.rerun()

# ------------------------------ GIAO DIỆN CHÍNH ---------------------------
st.markdown(
    f"""
    <div class="hero">
      <h1>🍜 {BOT_NAME}</h1>
      <p>Trợ lý AI của nhà hàng <b>{RESTAURANT}</b> — hỏi mình bất cứ điều gì về thực đơn nhé!</p>
    </div>
    """,
    unsafe_allow_html=True,
)

WELCOME = (
    f"Xin chào quý khách! Mình là **{BOT_NAME}**, trợ lý AI của nhà hàng {RESTAURANT}. "
    "Quý khách muốn tìm hiểu món nào ạ?"
)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": WELCOME}]

# Hiển thị lịch sử
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍🍳" if msg["role"] == "assistant" else "🙂"):
        st.markdown(msg["content"])

# Gợi ý câu hỏi (chỉ hiện khi mới bắt đầu)
clicked = None
if len(st.session_state.messages) == 1:
    st.caption("💡 Thử hỏi:")
    cols = st.columns(2)
    for i, s in enumerate(SUGGESTIONS):
        if cols[i % 2].button(s, key=f"sg{i}", width="stretch"):
            clicked = s.split(" ", 1)[1]

# Ô nhập câu hỏi
question = st.chat_input("Nhập câu hỏi của bạn...") or clicked

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🙂"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🧑‍🍳"):
        history = st.session_state.messages[1:]  # bỏ tin chào đầu tiên
        answer = st.write_stream(stream_answer(history, model_name, api_key, use_mock))
    st.session_state.messages.append({"role": "assistant", "content": answer})