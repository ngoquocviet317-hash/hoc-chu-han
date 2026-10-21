import os
import io
import json
import base64
import streamlit as st
import pandas as pd
from PIL import Image
from datetime import datetime
from google import genai
from google.genai import types
from st_img_pastebutton import paste

st.set_page_config(page_title="Sổ Tay Từ Vựng & Chiết Tự Hán Ngữ", page_icon="🈶", layout="wide")

DATA_FILE = "hanzi_vault.csv"

def compress_and_resize_image(input_img: Image.Image, max_dim: int = 1000) -> bytes:
    """Tự động resize và nén ảnh thành chuỗi byte JPEG nhẹ để truyền siêu tốc."""
    if input_img.mode in ("RGBA", "P"):
        input_img = input_img.convert("RGB")
    width, height = input_img.size
    if max(width, height) > max_dim:
        scale = max_dim / float(max(width, height))
        new_width = int(width * scale)
        new_height = int(height * scale)
        input_img = input_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    buffer = io.BytesIO()
    input_img.save(buffer, format="JPEG", quality=75, optimize=True)
    return buffer.getvalue()

def load_data():
    if os.path.exists(DATA_FILE):
        try:
            return pd.read_csv(DATA_FILE)
        except Exception:
            pass
    return pd.DataFrame(columns=[
        "word", "word_pinyin", "word_sino_vietnamese", "word_meaning", 
        "characters_detail_json", "created_at"
    ])

def save_or_update_words(new_entries: list[dict]):
    df = load_data()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    records_dict = {}
    if not df.empty:
        for _, row in df.iterrows():
            w = str(row.get("word", "")).strip()
            if w:
                records_dict[w] = row.to_dict()

    updated_count = 0
    added_count = 0

    for item in new_entries:
        word = item.get("word", "").strip()
        if not word:
            continue

        detail_json = json.dumps(item.get("characters", []), ensure_ascii=False)

        new_record = {
            "word": word,
            "word_pinyin": item.get("word_pinyin", ""),
            "word_sino_vietnamese": item.get("word_sino_vietnamese", ""),
            "word_meaning": item.get("word_meaning", ""),
            "characters_detail_json": detail_json,
            "created_at": now_str
        }

        if word in records_dict:
            updated_count += 1
        else:
            added_count += 1
            
        records_dict[word] = new_record

    new_df = pd.DataFrame(list(records_dict.values()))
    new_df.to_csv(DATA_FILE, index=False)
    return added_count, updated_count

def delete_single_word(word_to_delete: str):
    df = load_data()
    if not df.empty:
        df = df[df["word"] != word_to_delete]
        df.to_csv(DATA_FILE, index=False)

tab_scan, tab_vault = st.tabs(["⚡ Quét & Bóc Tách Từ Vựng", "📚 Sổ Tay Từ Đã Lưu"])

with tab_scan:
    st.sidebar.header("Cài Đặt")
    api_key = st.sidebar.text_input("Dán Gemini API Key vào đây:", type="password")

    st.subheader("📥 Nhập ảnh danh sách từ vựng")
    
    paste_data = paste(
        label="📋 BẤM VÀO ĐÂY ĐỂ DÁN ẢNH TỪ CLIPBOARD (Snipping Tool)",
        key="image_clipboard"
    )

    raw_img = None
    if paste_data is not None:
        try:
            header, encoded = paste_data.split(",", 1)
            image_bytes = base64.b64decode(encoded)
            raw_img = Image.open(io.BytesIO(image_bytes))
        except Exception as err:
            st.error(f"Lỗi đọc ảnh từ clipboard: {err}")

    with st.expander("Hoặc tải file ảnh từ máy tính (nếu cần)"):
        uploaded_file = st.file_uploader("Chọn tệp ảnh:", type=["jpg", "jpeg", "png", "webp"])
        if uploaded_file is not None:
            raw_img = Image.open(uploaded_file)

    if raw_img and api_key:
        img_bytes = compress_and_resize_image(raw_img)

        col_img, col_res = st.columns([1, 2])
        with col_img:
            st.image(raw_img, caption="Ảnh quét từ vựng", use_container_width=True)

        if st.button("🚀 Bắt Đầu Phân Tích Cụm Từ & Bộ Thủ", type="primary"):
            with st.spinner("AI đang nhận diện nguyên từ và chiết tự từng chữ..."):
                try:
                    client = genai.Client(api_key=api_key)
                    
                    prompt = """
                    Bạn là một chuyên gia Hán ngữ và phương pháp chiết tự chữ Hán.
                    Hãy quét toàn bộ từ vựng (từ đơn hoặc từ ghép nhiều chữ Hán) có trong bức ảnh.
                    Ví dụ nếu trong ảnh có từ ghép như "不客气", hãy nhận diện cả cụm từ trước, sau đó bóc tách từng chữ cấu thành.
                    
                    Trả về DUY NHẤT một mảng JSON (không có markdown thừa) theo cấu trúc sau:
                    [
                      {
                        "word": "từ vựng gốc đầy đủ (ví dụ: 不客气 hoặc 学习)",
                        "word_pinyin": "phiên âm pinyin của cả cụm (ví dụ: bú kèqi)",
                        "word_sino_vietnamese": "âm Hán Việt cả cụm (ví dụ: Bất Khách Khí)",
                        "word_meaning": "nghĩa tiếng Việt đầy đủ của từ (ví dụ: Đừng khách sáo, không có chi)",
                        "characters": [
                          {
                            "char": "不",
                            "char_pinyin": "bù / bú",
                            "char_hanviet": "Bất",
                            "char_meaning": "không, chẳng",
                            "components": [
                              {
                                "symbol": "一",
                                "name": "Bộ Nhất",
                                "role": "Biểu thị mặt đất hoặc đường ngang bầu trời"
                              }
                            ],
                            "mnemonics": "Mẹo nhớ chữ 不: Hình ảnh chú chim bay vút lên trời cao nhưng chưa tới nghĩa là chưa được, không được."
                          },
                          {
                            "char": "客",
                            "char_pinyin": "kè",
                            "char_hanviet": "Khách",
                            "char_meaning": "khách khứa, khách trọ",
                            "components": [
                              {
                                "symbol": "宀",
                                "name": "Bộ Miên",
                                "role": "Mái nhà"
                              },
                              {
                                "symbol": "夂",
                                "name": "Bộ Trĩ",
                                "role": "Đến sau, bước chân đến"
                              },
                              {
                                "symbol": "口",
                                "name": "Bộ Khẩu",
                                "role": "Miệng"
                              }
                            ],
                            "mnemonics": "Mẹo nhớ chữ 客: Người bước chân vào dưới mái nhà mở miệng chào hỏi chính là KHÁCH."
                          },
                          {
                            "char": "气",
                            "char_pinyin": "qì",
                            "char_hanviet": "Khí",
                            "char_meaning": "khí hậu, hơi thở, thái độ",
                            "components": [
                              {
                                "symbol": "气",
                                "name": "Bộ Khí",
                                "role": "Luồng mây, hơi nước bốc lên"
                              }
                            ],
                            "mnemonics": "Mẹo nhớ chữ 气: Ba luồng hơi nước uốn lượn bốc lên trời thành luồng KHÍ."
                          }
                        ]
                      }
                    ]
                    """
