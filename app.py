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
    cols = ["word", "word_pinyin", "word_sino_vietnamese", "word_meaning", "characters_detail_json", "created_at"]
    if os.path.exists(DATA_FILE):
        try:
            df = pd.read_csv(DATA_FILE)
            if "character" in df.columns and "word" not in df.columns:
                df["word"] = df["character"]
            for c in cols:
                if c not in df.columns:
                    df[c] = ""
            return df[cols]
        except Exception:
            pass
    return pd.DataFrame(columns=cols)

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
            with st.spinner("AI đang nhận diện nguyên từ và chiết tự từng bộ thủ..."):
                try:
                    client = genai.Client(api_key=api_key)
                    
                    prompt_text = """
                    Bạn là một chuyên gia Hán ngữ và chiết tự chữ Hán.
                    Hãy nhận diện toàn bộ từ vựng (từ đơn hoặc từ ghép nhiều chữ Hán) có trong ảnh.
                    Với mỗi từ, hãy nhận diện nghĩa cả từ, sau đó BẮT BUỘC bóc tách chi tiết cấu tạo từng chữ cấu thành:
                    - Từng chữ gồm những bộ thủ / nét thành phần nào: ký hiệu bộ, tên Hán Việt của bộ, ý nghĩa biểu thị của bộ.
                    - Mẹo ghi nhớ sinh động liên kết các bộ thủ với nghĩa của chữ đó.

                    Hãy trả về DUY NHẤT một chuỗi JSON thuần (không có markdown ```json):
                    [
                      {
                        "word": "学生",
                        "word_pinyin": "xuéshēng",
                        "word_sino_vietnamese": "Học Sinh",
                        "word_meaning": "học sinh, học viên",
                        "characters": [
                          {
                            "char": "学",
                            "char_pinyin": "xué",
                            "char_hanviet": "Học",
                            "char_meaning": "học tập, bắt chước",
                            "components": [
                              {"symbol": "⺍", "name": "Bộ Hào biến thể", "role": "Tia sáng tri thức, bắt chước"},
                              {"symbol": "冖", "name": "Bộ Mịch", "role": "Mái che, lớp phủ"},
                              {"symbol": "子", "name": "Bộ Tử", "role": "Đứa trẻ, con cái"}
                            ],
                            "mnemonics": "Đứa trẻ ngồi dưới mái che tiếp thu tri thức chính là việc Học."
                          },
                          {
                            "char": "生",
                            "char_pinyin": "shēng",
                            "char_hanviet": "Sinh",
                            "char_meaning": "sinh ra, mầm sống",
                            "components": [
                              {"symbol": "生", "name": "Bộ Sinh", "role": "Mầm sống đâm chồi"}
                            ],
                            "mnemonics": "Mầm cây đâm chồi nhú lên mặt đất là sự Sinh sôi."
                          }
                        ]
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

                            st.markdown(f"### 🈴 Chữ **`{c_char}`** (Pinyin: *{c_py}* | Hán Việt: *{c_hv}*): {c_mean}")
                            st.markdown("**🧩 Bộ thủ cấu tạo:**")
                            if isinstance(c_comps, list) and len(c_comps) > 0:
                                for comp in c_comps:
                                    if isinstance(comp, dict):
                                        sym = comp.get("symbol", "")
                                        cname = comp.get("name", "")
                                        crole = comp.get("role", "")
                                        st.markdown(f"- **{sym}** ({cname}): *{crole}*")
                                    else:
                                        st.markdown(f"- {comp}")
                            else:
                                st.markdown("- *Bộ thủ nguyên bản*")

                            if c_mnem:
                                st.info(f"💡 **Mẹo ghi nhớ chữ {c_char}:** {c_mnem}")
                            st.divider()

    elif not api_key:
        st.info("Vui lòng dán API Key vào thanh bên trái để sử dụng.")

with tab_vault:
    df_saved = load_data()
    st.subheader(f"Tổng số từ vựng đang lưu: {len(df_saved)}")

    if not df_saved.empty:
        col_search, col_clear = st.columns([3, 1])
        with col_search:
            search_kw = st.text_input("🔍 Tra cứu (chữ Hán, nghĩa tiếng Việt, Pinyin, Hán Việt):").strip().lower()
        with col_clear:
            st.write("")
            st.write("")
            if st.button("🗑️ Xóa toàn bộ sổ tay"):
                if os.path.exists(DATA_FILE):
                    os.remove(DATA_FILE)
                st.success("Đã xóa sạch sổ tay!")
                st.rerun()

        if search_kw:
            matched_indices = []
            for idx, row in df_saved.iterrows():
                full_search_text = (
                    str(row.get("word", "")) + " " +
                    str(row.get("word_meaning", "")) + " " +
                    str(row.get("word_pinyin", "")) + " " +
                    str(row.get("word_sino_vietnamese", "")) + " " +
                    str(row.get("characters_detail_json", ""))
                ).lower()
                if search_kw in full_search_text:
                    matched_indices.append(idx)
            df_display = df_saved.loc[matched_indices]
        else:
            df_display = df_saved

        st.caption(f"Tìm thấy {len(df_display)} từ phù hợp:")

        for idx, row in df_display.iterrows():
            w_text = row.get("word", "")
            w_py = row.get("word_pinyin", "")
            w_hv = row.get("word_sino_vietnamese", "")
            w_mean = row.get("word_meaning", "")
            save_time = str(row.get("created_at", "Chưa rõ"))
            details_raw = row.get("characters_detail_json", "[]")
            try:
                char_list = json.loads(details_raw)
            except Exception:
                char_list = []

            with st.container():
                col_info, col_del = st.columns([5, 1])
                with col_info:
                    st.markdown(f"### 🈶 {w_text} `[{w_py}]` — **{w_hv}**")
                    st.markdown(f"🎯 **Nghĩa tiếng Việt:** **{w_mean}**")
                    if char_list:
                        with st.expander("👉 Bấm để xem chi tiết chiết tự từng chữ cấu thành:"):
                            for c in char_list:
                                c_char = c.get("char", "")
                                c_py = c.get("char_pinyin", "")
                                c_hv = c.get("char_hanviet", "")
                                c_mean = c.get("char_meaning", "")
                                c_comps = c.get("components", [])
                                c_mnem = c.get("mnemonics", "")
                                st.markdown(f"**Chữ `{c_char}`** (*{c_py}* - *{c_hv}*): {c_mean}")
                                for comp in c_comps:
                                    if isinstance(comp, dict):
                                        st.markdown(f"- **{comp.get('symbol', '')}** ({comp.get('name', '')}): {comp.get('role', '')}")
                                    else:
                                        st.markdown(f"- {comp}")
                                if c_mnem:
                                    st.caption(f"💡 Mẹo nhớ: {c_mnem}")
                                st.write("---")
                    st.caption("Cập nhật lúc: " + save_time)
                with col_del:
                    st.write("")
                    st.write("")
                    if st.button("Xóa từ này", key=f"del_{w_text}_{idx}"):
                        delete_single_word(w_text)
                        st.rerun()
                st.divider()

        csv_data = df_saved.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 Tải toàn bộ sổ tay về máy (.CSV mở Excel)",
            data=csv_data,
            file_name="so_tay_tu_vung_han_ngu.csv",
            mime="text/csv"
        )
    else:
        st.write("Chưa có từ vựng nào được lưu.")
