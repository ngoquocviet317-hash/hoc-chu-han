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

st.set_page_config(page_title="Sổ Tay Hán Tự", page_icon="🈶", layout="wide")

DATA_FILE = "hanzi_vault.csv"

def compress_and_resize_image(input_img: Image.Image, max_dim: int = 1000) -> bytes:
    """Resize và nén ảnh thành chuỗi byte JPEG nhẹ để truyền siêu tốc."""
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
    return pd.DataFrame(columns=["character", "pinyin", "sino_vietnamese", "meaning", "components", "mnemonics", "created_at"])

def save_new_words(new_entries: list[dict]):
    df = load_data()
    existing_chars = set(df["character"].dropna().tolist())
    records_to_add = []
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    for item in new_entries:
        char = item.get("character", "").strip()
        if char and char not in existing_chars:
            comps = item.get("components", [])
            comp_str = ", ".join(comps) if isinstance(comps, list) else str(comps)
            records_to_add.append({
                "character": char,
                "pinyin": item.get("pinyin", ""),
                "sino_vietnamese": item.get("sino_vietnamese", ""),
                "meaning": item.get("meaning", ""),
                "components": comp_str,
                "mnemonics": item.get("mnemonics", ""),
                "created_at": now_str
            })
            existing_chars.add(char)

    if records_to_add:
        updated_df = pd.concat([df, pd.DataFrame(records_to_add)], ignore_index=True)
        updated_df.to_csv(DATA_FILE, index=False)
        return len(records_to_add)
    return 0

tab_scan, tab_vault = st.tabs(["⚡ Quét & Tách Chữ Nhanh", "📚 Sổ Tay Đã Lưu"])

with tab_scan:
    st.sidebar.header("Cài Đặt")
    api_key = st.sidebar.text_input("Dán Gemini API Key vào đây:", type="password")

    st.subheader("📥 Nhập ảnh chữ Hán từ Snipping Tool")
    
    paste_data = paste(
        label="📋 BẤM VÀO ĐÂY ĐỂ DÁN ẢNH TỪ CLIPBOARD",
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

    with st.expander("Hoặc tải file ảnh từ máy"):
        uploaded_file = st.file_uploader("Chọn ảnh:", type=["jpg", "jpeg", "png", "webp"])
        if uploaded_file is not None:
            raw_img = Image.open(uploaded_file)

    if raw_img and api_key:
        img_bytes = compress_and_resize_image(raw_img)

        col_img, col_res = st.columns([1, 2])
        with col_img:
            st.image(raw_img, caption="Ảnh chữ Hán cần phân tích", use_container_width=True)

        if st.button("🚀 Bắt Đầu Phân Tích", type="primary"):
            with st.spinner("AI đang nhận diện siêu tốc..."):
                try:
                    client = genai.Client(api_key=api_key)
                    
                    prompt = """
                    Quét toàn bộ chữ Hán có trong bức ảnh.
                    Hãy trả về DUY NHẤT một mảng JSON (không bọc mã markdown thừa) theo cấu trúc sau:
                    [
                      {
                        "character": "chữ Hán",
                        "pinyin": "phiên âm pinyin",
                        "sino_vietnamese": "âm Hán Việt",
                        "meaning": "nghĩa tiếng Việt ngắn gọn",
                        "components": ["bộ thủ 1", "bộ thủ 2"],
                        "mnemonics": "mẹo nhớ ngắn gọn liên kết các bộ với nghĩa"
                      }
                    ]
                    """
                    
                    models_to_try = [
                        "gemini-2.0-flash",
                        "gemini-2.5-flash",
                        "gemini-1.5-pro",
                        "gemini-1.5-flash"
                    ]
                    
                    response = None
                    last_err = None
                    for m in models_to_try:
                        try:
                            response = client.models.generate_content(
                                model=m,
                                contents=[
                                    types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                                    prompt
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
                    
                    st.session_state["scanned_items"] = parsed_data
                    st.rerun()
                except Exception as e:
                    st.error(f"Lỗi xử lý: {e}")

        if "scanned_items" in st.session_state:
            items = st.session_state["scanned_items"]
            with col_res:
                st.subheader(f"⚡ Đã quét xong: {len(items)} chữ")
                if st.button("💾 Lưu tất cả vào sổ tay", type="secondary"):
                    added = save_new_words(items)
                    st.success(f"Đã lưu thành công {added} chữ mới!")

                for item in items:
                    char = item.get("character", "")
                    pinyin = item.get("pinyin", "")
                    hanviet = item.get("sino_vietnamese", "")
                    meaning = item.get("meaning", "")
                    comps = item.get("components", [])
                    comp_text = ", ".join(comps) if isinstance(comps, list) else str(comps)
                    mnem = item.get("mnemonics", "")
                    
                    with st.expander(f"**{char}** [{pinyin}] — {hanviet}: {meaning}", expanded=True):
                        st.markdown(f"**Bộ thủ:** {comp_text}")
                        st.info(f"💡 **Mẹo nhớ:** {mnem}")

    elif not api_key:
        st.info("Vui lòng dán API Key vào thanh bên trái để sử dụng.")

with tab_vault:
    df_saved = load_data()
    st.subheader(f"Tổng số từ đang lưu: {len(df_saved)}")

    if not df_saved.empty:
        search_kw = st.text_input("Tìm kiếm từ (chữ Hán, Pinyin, Hán Việt):").strip().lower()
        if search_kw:
            df_display = df_saved[
                df_saved["character"].astype(str).str.contains(search_kw, case=False, na=False) |
                df_saved["pinyin"].astype(str).str.contains(search_kw, case=False, na=False) |
                df_saved["sino_vietnamese"].astype(str).str.contains(search_kw, case=False, na=False) |
                df_saved["meaning"].astype(str).str.contains(search_kw, case=False, na=False)
            ]
        else:
            df_display = df_saved

        for _, row in df_display.iterrows():
            st.markdown(f"### {row['character']} `/{row['pinyin']}/` — **{row['sino_vietnamese']}**")
            st.markdown(f"- **Nghĩa:** {row['meaning']}")
            st.markdown(f"- **Bộ thủ:** {row['components']}")
            st.markdown(f"- **Mẹo nhớ:** {row['mnemonics']}")
            st.caption(f"Lưu lúc: {row.get('created_at', 'Chưa rõ')}")
            st.divider()

        csv_data = df_saved.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="Tải file CSV sổ tay về máy",
            data=csv_data,
            file_name="so_tay_chu_han.csv",
            mime="text/csv"
        )
    else:
        st.write("Chưa có từ nào được lưu.")
