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
    columns = ["word", "word_pinyin", "word_sino_vietnamese", "word_meaning", "characters_detail_json", "created_at"]
    if os.path.exists(DATA_FILE):
        try:
            df = pd.read_csv(DATA_FILE)
            # Tự động đồng bộ nếu là file từ các phiên bản cũ
            if "character" in df.columns and "word" not in df.columns:
                df["word"] = df["character"]
            for col in columns:
                if col not in df.columns:
                    df[col] = ""
            return df[columns]
        except Exception:
            pass
    return pd.DataFrame(columns=columns)

def save_or_update_words(new_entries: list):
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

    with st.expander("Hoặc tải file ảnh từ máy tính"):
        uploaded_file = st.file_uploader("Chọn tệp ảnh:", type=["jpg", "jpeg", "png", "webp"])
        if uploaded_file is not None:
            raw_img = Image.open(uploaded_file)

    if raw_img and api_key:
        img_bytes = compress_and_resize_image(raw_img)
        col_img, col_res = st.columns([1, 2])
        with col_img:
            st.image(raw_img, caption="Ảnh quét từ vựng", use_container_width=True)

        if st.button("🚀 Bắt Đầu Phân Tích", type="primary"):
            with st.spinner("AI đang nhận diện nguyên từ và chiết tự từng chữ..."):
                try:
                    client = genai.Client(api_key=api_key)
                    prompt_text = (
                        "Bạn là chuyên gia Hán ngữ và chiết tự chữ Hán. "
                        "Hãy quét toàn bộ từ vựng (từ đơn hoặc từ ghép nhiều chữ Hán) có trong bức ảnh. "
                        "Ví dụ nếu có từ '不客气', hãy nhận diện nguyên từ trước rồi bóc tách từng chữ cấu thành. "
                        "Trả về duy nhất mảng JSON không bọc markdown theo cấu trúc: "
                        "[{\"word\": \"từ gốc\", \"word_pinyin\": \"pinyin\", \"word_sino_vietnamese\": \"Hán Việt\", "
                        "\"word_meaning\": \"nghĩa tiếng Việt đầy đủ\", \"characters\": [{\"char\": \"chữ\", "
                        "\"char_pinyin\": \"pinyin\", \"char_hanviet\": \"Hán Việt\", \"char_meaning\": \"nghĩa\", "
                        "\"components\": [{\"symbol\": \"bộ\", \"name\": \"tên bộ\", \"role\": \"ý nghĩa bộ\"}], "
                        "\"mnemonics\": \"mẹo nhớ ngắn gọn liên kết bộ thủ với nghĩa\"}]}]"
                    )

                    # Tự động thử lần lượt các model sẵn có, không cần người dùng chọn thủ công
                    models_to_try = [
                        "gemini-3.5-flash-lite",
                        "gemini-3.8-flash",
                        "gemini-3.1-pro"
                    ]

                    response = None
                    last_err = None
                    for m in models_to_try:
                        try:
                            response = client.models.generate_content(
                                model=m,
                                contents=[
                                    types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                                    prompt_text
                                ],
                                config=types.GenerateContentConfig(
                                    response_mime_type="application/json",
                                    temperature=0.1
                                )
                            )
                            if response and response.text:
                                break
                        except Exception as e:
                            last_err = e
                            continue

                    if not response or not response.text:
                        raise Exception(f"Không thể kết nối máy chủ AI: {last_err}")

                    clean_text = response.text.strip()
                    if clean_text.startswith("```json"):
                        clean_text = clean_text[7:]
                    if clean_text.startswith("```"):
                        clean_text = clean_text[3:]
                    if clean_text.endswith("```"):
                        clean_text = clean_text[:-3]

                    parsed_data = json.loads(clean_text.strip())
                    if isinstance(parsed_data, dict) and "words" in parsed_data:
                        parsed_data = parsed_data["words"]

                    st.session_state["scanned_words"] = parsed_data
                    st.rerun()
                except Exception as e:
                    st.error(f"Lỗi xử lý: {e}")

        if "scanned_words" in st.session_state:
            words = st.session_state["scanned_words"]
            with col_res:
                st.subheader(f"⚡ Nhận diện được {len(words)} từ vựng")
                if st.button("💾 Lưu vào sổ tay (Tự động ghi đè từ đã có)", type="secondary"):
                    added, updated = save_or_update_words(words)
                    msg = []
                    if added > 0:
                        msg.append(f"thêm mới {added} từ")
                    if updated > 0:
                        msg.append(f"ghi đè {updated} từ cũ")
                    st.success(f"Đã {' và '.join(msg)} vào sổ tay!")

                for w_item in words:
                    w_text = w_item.get("word", "")
                    w_py = w_item.get("word_pinyin", "")
                    w_hv = w_item.get("word_sino_vietnamese", "")
                    w_mean = w_item.get("word_meaning", "")
                    char_list = w_item.get("characters", [])

                    with st.expander(f"✨ **{w_text}** `[{w_py}]` — **{w_hv}**: {w_mean}", expanded=True):
                        st.markdown(f"#### 📖 Nghĩa của từ: **{w_mean}**")
                        st.markdown("---")
                        st.markdown("##### 🔍 Bóc tách từng chữ cấu thành:")
                        for c in char_list:
                            c_char = c.get("char", "")
                            c_py = c.get("char_pinyin", "")
                            c_hv = c.get("char_hanviet", "")
                            c_mean = c.get("char_meaning", "")
                            c_comps = c.get("components", [])
                            c_mnem = c.get("mnemonics", "")

                            st.markdown(f"**Chữ `{c_char}`** (Pinyin: *{c_py}* | Hán Việt: *{c_hv}*): {c_mean}")
                            comp_lines = []
                            for comp in c_comps:
                                if isinstance(comp, dict):
                                    comp_lines.append(f"  * **{comp.get('symbol', '')}** ({comp.get('name', '')}): {comp.get('role', '')}")
