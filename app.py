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

st.set_page_config(page_title="Sổ Tay Hán Tự & Chiết Tự", page_icon="🈶", layout="wide")

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
    return pd.DataFrame(columns=["character", "pinyin", "sino_vietnamese", "meaning", "components", "mnemonics", "created_at"])

# --- HÀM LƯU TỪ VỰNG: GHI ĐÈ TỪ ĐÃ CÓ VÀ THÊM TỪ MỚI ---
def save_or_update_words(new_entries: list[dict]):
    df = load_data()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    # Chuyển DataFrame hiện tại thành dictionary để dễ tra cứu và ghi đè theo chữ Hán
    records_dict = {}
    if not df.empty:
        for _, row in df.iterrows():
            char = str(row["character"]).strip()
            if char:
                records_dict[char] = row.to_dict()

    updated_count = 0
    added_count = 0

    for item in new_entries:
        char = item.get("character", "").strip()
        if not char:
            continue

        comps = item.get("components", [])
        if isinstance(comps, list):
            comp_parts = []
            for c in comps:
                if isinstance(c, dict):
                    sym = c.get("symbol", "")
                    name = c.get("name", "")
                    role = c.get("role", "")
                    comp_parts.append(f"{sym} - {name} ({role})")
                else:
                    comp_parts.append(str(c))
            comp_str = " | ".join(comp_parts)
        else:
            comp_str = str(comps)

        new_record = {
            "character": char,
            "pinyin": item.get("pinyin", ""),
            "sino_vietnamese": item.get("sino_vietnamese", ""),
            "meaning": item.get("meaning", ""),
            "components": comp_str,
            "mnemonics": item.get("mnemonics", ""),
            "created_at": now_str
        }

        if char in records_dict:
            updated_count += 1
        else:
            added_count += 1
            
        # Ghi đè hoặc thêm mới
        records_dict[char] = new_record

    # Lưu lại toàn bộ vào file CSV
    new_df = pd.DataFrame(list(records_dict.values()))
    new_df.to_csv(DATA_FILE, index=False)
    return added_count, updated_count

# Hàm xóa một từ cụ thể
def delete_single_word(char_to_delete: str):
    df = load_data()
    if not df.empty:
        df = df[df["character"] != char_to_delete]
        df.to_csv(DATA_FILE, index=False)

tab_scan, tab_vault = st.tabs(["⚡ Quét & Chiết Tự", "📚 Sổ Tay Đã Lưu"])

with tab_scan:
    st.sidebar.header("Cài Đặt")
    api_key = st.sidebar.text_input("Dán Gemini API Key vào đây:", type="password")

    st.subheader("📥 Nhập ảnh chữ Hán từ Snipping Tool")
    
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
            st.image(raw_img, caption="Ảnh chữ Hán cần phân tích", use_container_width=True)

        if st.button("🚀 Bắt Đầu Phân Tích Chi Tiết", type="primary"):
            with st.spinner("AI đang nhận diện và bóc tách bộ thủ siêu tốc..."):
                try:
                    client = genai.Client(api_key=api_key)
                    
                    prompt = """
                    Bạn là một chuyên gia chiết tự và ngữ nghĩa học chữ Hán.
                    Hãy quét toàn bộ chữ Hán có trong bức ảnh này và bóc tách cấu tạo cực kỳ chi tiết.
                    Trả về DUY NHẤT một mảng JSON (không bọc mã markdown thừa) theo cấu trúc sau:
                    [
                      {
                        "character": "chữ Hán cần phân tích (ví dụ: 汉)",
                        "pinyin": "phiên âm pinyin kèm thanh điệu (ví dụ: hàn)",
                        "sino_vietnamese": "âm Hán Việt (ví dụ: Hán)",
                        "meaning": "nghĩa cơ bản của chữ",
                        "components": [
                          {
                            "symbol": "氵",
                            "name": "Bộ Thủy (Ba chấm thủy)",
                            "role": "Biểu thị nước, sông suối (nguyên gốc chỉ sông Hán Thủy)"
                          },
                          {
                            "symbol": "又",
                            "name": "Bộ Hựu",
                            "role": "Bàn tay phải, động tác lặp đi lặp lại"
                          }
                        ],
                        "mnemonics": "Mẹo kể chuyện ngắn gọn, sinh động liên kết trực tiếp ý nghĩa của các bộ thủ ở trên với nghĩa tổng thể của chữ để người học nhớ sâu"
                      }
                    ]
                    """
                    
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
                
                # Nút lưu với cơ chế Ghi Đè thông minh
                if st.button("💾 Lưu vào sổ tay (Tự động ghi đè từ đã có)", type="secondary"):
                    added, updated = save_or_update_words(items)
                    msg = []
                    if added > 0:
                        msg.append(f"thêm mới {added} từ")
                    if updated > 0:
                        msg.append(f"ghi đè thành công {updated} từ cũ")
                    st.success(f"Đã {' và '.join(msg)} vào sổ tay!")

                for item in items:
                    char = item.get("character", "")
                    pinyin = item.get("pinyin", "")
                    hanviet = item.get("sino_vietnamese", "")
                    meaning = item.get("meaning", "")
                    comps = item.get("components", [])
                    mnem = item.get("mnemonics", "")
                    
                    with st.expander(f"🈶 **{char}** `/{pinyin}/` — **{hanviet}**: {meaning}", expanded=True):
                        st.markdown("##### 🧩 Thành phần & Bộ thủ cấu tạo:")
                        if isinstance(comps, list):
                            for c in comps:
                                if isinstance(c, dict):
                                    sym = c.get("symbol", "")
                                    cname = c.get("name", "")
                                    crole = c.get("role", "")
                                    st.markdown(f"- **{sym}** ({cname}): *{crole}*")
                                else:
                                    st.markdown(f"- {c}")
                        else:
                            st.markdown(f"- {comps}")
                            
                        st.info(f"💡 **Mẹo ghi nhớ (Mnemonics):**\n\n{mnem}")

    elif not api_key:
        st.info("Vui lòng dán API Key vào thanh bên trái để sử dụng.")

with tab_vault:
    df_saved = load_data()
    st.subheader(f"Tổng số từ đang lưu: {len(df_saved)}")

    if not df_saved.empty:
        col_search, col_clear = st.columns([3, 1])
        with col_search:
            search_kw = st.text_input("Tìm kiếm từ (chữ Hán, Pinyin, Hán Việt):").strip().lower()
        with col_clear:
            st.write("")
            st.write("")
            # Nút xóa trắng toàn bộ sổ tay nếu muốn làm lại từ đầu
            if st.button("🗑️ Xóa toàn bộ sổ tay"):
                if os.path.exists(DATA_FILE):
                    os.remove(DATA_FILE)
                st.success("Đã xóa sạch sổ tay!")
                st.rerun()

        if search_kw:
            df_display = df_saved[
                df_saved["character"].astype(str).str.contains(search_kw, case=False, na=False) |
                df_saved["pinyin"].astype(str).str.contains(search_kw, case=False, na=False) |
                df_saved["sino_vietnamese"].astype(str).str.contains(search_kw, case=False, na=False) |
                df_saved["meaning"].astype(str).str.contains(search_kw, case=False, na=False)
            ]
        else:
            df_display = df_saved

        for idx, row in df_display.iterrows():
            char = row['character']
            with st.container():
                col_info, col_del = st.columns([5, 1])
                with col_info:
                    st.markdown(f"### {char} `/{row['pinyin']}/` — **{row['sino_vietnamese']}**")
                    st.markdown(f"- **Nghĩa:** {row['meaning']}")
                    st.markdown(f"- **Bộ thủ:** {row['components']}")
                    st.markdown(f"- **Mẹo nhớ:** {row['mnemonics']}")
                    st.caption(f"Cập nhật lúc: {row.get('created_at', 'Chưa rõ')}")
                with col_del:
                    st.write("")
                    st.write("")
                    if st.button("Xóa chữ này", key=f"del_{char}_{idx}"):
                        delete_single_word(char)
                        st.rerun()
                st.divider()

        csv_data = df_saved.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 Tải file CSV sổ tay về máy",
            data=csv_data,
            file_name="so_tay_chu_han.csv",
            mime="text/csv"
        )
    else:
        st.write("Chưa có từ nào được lưu.")
