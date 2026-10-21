import os
import io
import streamlit as st
import pandas as pd
from PIL import Image
from datetime import datetime
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from streamlit_paste_button import paste_image_button

st.set_page_config(page_title="Sổ Tay Hán Tự", page_icon="🈶", layout="wide")

DATA_FILE = "hanzi_vault.csv"

# --- HÀM NÉN ẢNH SIÊU TỐC ---
def compress_and_resize_image(input_img: Image.Image, max_dim: int = 1200) -> Image.Image:
    """Tự động resize và tối ưu ảnh để xử lý siêu nhanh mà vẫn giữ sắc nét chữ Hán."""
    if input_img.mode in ("RGBA", "P"):
        input_img = input_img.convert("RGB")
    
    width, height = input_img.size
    if max(width, height) > max_dim:
        scale = max_dim / float(max(width, height))
        new_width = int(width * scale)
        new_height = int(height * scale)
        input_img = input_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    buffer = io.BytesIO()
    input_img.save(buffer, format="JPEG", quality=80, optimize=True)
    buffer.seek(0)
    return Image.open(buffer)

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
        if item["character"] not in existing_chars:
            records_to_add.append({
                "character": item["character"],
                "pinyin": item["pinyin"],
                "sino_vietnamese": item["sino_vietnamese"],
                "meaning": item["meaning"],
                "components": ", ".join(item["components"]) if isinstance(item["components"], list) else item["components"],
                "mnemonics": item["mnemonics"],
                "created_at": now_str
            })
            existing_chars.add(item["character"])

    if records_to_add:
        updated_df = pd.concat([df, pd.DataFrame(records_to_add)], ignore_index=True)
        updated_df.to_csv(DATA_FILE, index=False)
        return len(records_to_add)
    return 0

class HanziDetail(BaseModel):
    character: str = Field(description="Chữ Hán chuẩn")
    pinyin: str = Field(description="Pinyin kèm dấu")
    sino_vietnamese: str = Field(description="Âm Hán Việt")
    meaning: str = Field(description="Nghĩa ngắn gọn")
    components: list[str] = Field(description="Các bộ thủ cấu thành")
    mnemonics: str = Field(description="Mẹo liên tưởng dễ nhớ")

class OCRAnalysisResult(BaseModel):
    words: list[HanziDetail]

tab_scan, tab_vault = st.tabs(["⚡ Quét & Tách Chữ Nhanh", "📚 Sổ Tay Đã Lưu"])

with tab_scan:
    st.sidebar.header("Cài Đặt")
    api_key = st.sidebar.text_input("Dán Gemini API Key vào đây:", type="password")

    st.subheader("📥 Nhập ảnh chữ Hán (Dán trực tiếp không cần lưu)")
    
    col_btn, col_alt = st.columns([1, 1])
    with col_btn:
        st.markdown("**Cách 1: Dán nhanh từ bộ nhớ tạm**")
        paste_result = paste_image_button(
            label="📋 Bấm vào đây để DÁN ẢNH vừa chụp (Snipping Tool)",
            background_color="#FF4B4B",
            hover_background_color="#E03838",
            key="paste_clipboard"
        )
    with col_alt:
        st.markdown("**Cách 2: Kéo thả file nếu có sẵn**")
        uploaded_file = st.file_uploader("Kéo thả file vào đây (tùy chọn):", type=["jpg", "jpeg", "png", "webp"])

    raw_img = None
    if paste_result.image_data is not None:
        raw_img = paste_result.image_data
    elif uploaded_file is not None:
        raw_img = Image.open(uploaded_file)

    if raw_img and api_key:
        # Nén ảnh ngay tại chỗ để đạt tốc độ cao nhất
        img = compress_and_resize_image(raw_img)

        col_img, col_res = st.columns([1, 2])
        with col_img:
            st.image(img, caption="Ảnh đã nhận & tối ưu dung lượng", use_container_width=True)

        if st.button("🚀 Bắt Đầu Phân Tích Siêu Tốc", type="primary"):
            with st.spinner("AI đang nhận diện và bóc tách bộ thủ..."):
                try:
                    client = genai.Client(api_key=api_key)
                    prompt = """
                    Quét toàn bộ chữ Hán có trong ảnh:
                    1. Nhận diện chuẩn xác từng chữ Hán.
                    2. Phân tích các bộ thủ tạo nên chữ.
                    3. Gợi ý mẹo nhớ ngắn gọn bằng tiếng Việt liên kết giữa bộ thủ và ý nghĩa.
                    Trả về đúng chuẩn schema.
                    """
                    response = client.models.generate_content(
                        model="gemini-1.5-flash",
                        contents=[img, prompt],
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=OCRAnalysisResult,
                            temperature=0.1
                        )
                    )
                    data = OCRAnalysisResult.model_validate_json(response.text)
                    st.session_state["scanned_items"] = [item.model_dump() for item in data.words]
                except Exception as e:
                    st.error(f"Lỗi: {e}")

        if "scanned_items" in st.session_state:
            items = st.session_state["scanned_items"]
            with col_res:
                st.subheader(f"⚡ Kết quả nhận diện: {len(items)} chữ")
                if st.button("💾 Lưu tất cả vào sổ tay", type="secondary"):
                    added = save_new_words(items)
                    st.success(f"Đã lưu thành công {added} chữ mới!")

                for item in items:
                    with st.expander(f"**{item['character']}** [{item['pinyin']}] — {item['sino_vietnamese']}: {item['meaning']}", expanded=True):
                        st.markdown(f"**Bộ thủ:** {', '.join(item['components'])}")
                        st.info(f"💡 **Mẹo nhớ:** {item['mnemonics']}")
    elif not api_key:
        st.info("Vui lòng dán API Key vào cột bên trái để bắt đầu.")

with tab_vault:
    df_saved = load_data()
    st.subheader(f"Tổng số từ đang lưu: {len(df_saved)}")

    if not df_saved.empty:
        search_kw = st.text_input("🔍 Tìm kiếm từ (chữ Hán, Pinyin, Hán Việt):").strip().lower()
        if search_kw:
            df_display = df_saved[
                df_saved["character"].str.contains(search_kw, na=False) |
                df_saved["pinyin"].str.lower().contains(search_kw, na=False) |
                df_saved["sino_vietnamese"].str.lower().contains(search_kw, na=False) |
                df_saved["meaning"].str.lower().contains(search_kw, na=False)
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
            label="📥 Tải file CSV sổ tay về máy",
            data=csv_data,
            file_name="so_tay_chu_han.csv",
            mime="text/csv"
        )
    else:
        st.write("Chưa có từ nào được lưu.")
