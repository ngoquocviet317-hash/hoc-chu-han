import os
import io
import base64
import streamlit as st
import pandas as pd
from PIL import Image
from datetime import datetime
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

st.set_page_config(page_title="Sổ Tay Hán Tự", page_icon="🈶", layout="wide")

DATA_FILE = "hanzi_vault.csv"

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
    pinyin: str = Field(description="Pinyin kèm dấu thanh điệu")
    sino_vietnamese: str = Field(description="Âm Hán Việt tương ứng")
    meaning: str = Field(description="Nghĩa ngắn gọn")
    components: list[str] = Field(description="Các bộ thủ cấu thành")
    mnemonics: str = Field(description="Mẹo liên tưởng dễ nhớ")

class OCRAnalysisResult(BaseModel):
    words: list[HanziDetail]

tab_scan, tab_vault = st.tabs(["📸 Nhận Diện & Phân Tích", "📚 Sổ Tay Đã Lưu"])

with tab_scan:
    st.sidebar.header("Cài Đặt")
    api_key = st.sidebar.text_input("Dán Gemini API Key vào đây:", type="password")

    st.subheader("📥 Nhập ảnh chữ Hán")

    # Nhúng vùng bắt sự kiện Paste trực tiếp bằng HTML & JS
    paste_html = """
    <div id="drop-box" style="
        border: 2px dashed #FF4B4B; 
        border-radius: 10px; 
        padding: 28px; 
        text-align: center; 
        background: rgba(255, 75, 75, 0.05); 
        cursor: pointer;
        font-family: sans-serif;
    " tabindex="0">
        <h4 style="margin: 0; color: #FF4B4B;">📋 Bấm chuột vào ô này rồi nhấn Ctrl + V để DÁN ẢNH</h4>
        <p style="margin: 8px 0 0 0; color: #888; font-size: 13px;">Hỗ trợ ảnh copy từ Snipping Tool, Zalo, hoặc trình duyệt</p>
    </div>

    <script>
    const box = document.getElementById('drop-box');
    box.addEventListener('paste', function(e) {
        const items = (e.clipboardData || e.originalEvent.clipboardData).items;
        for (let index in items) {
            const item = items[index];
            if (item.kind === 'file') {
                const blob = item.getAsFile();
                const reader = new FileReader();
                reader.onload = function(event) {
                    const base64Data = event.target.result;
                    // Đưa dữ liệu ảnh vào Streamlit session qua URL hash
                    window.parent.postMessage({
                        type: 'streamlit:setComponentValue',
                        value: base64Data
                    }, '*');
                };
                reader.readAsDataURL(blob);
                box.style.borderColor = '#00C853';
                box.innerHTML = '<h4 style="margin: 0; color: #00C853;">✅ Đã nhận được ảnh! Đang tải lên...</h4>';
            }
        }
    });
    </script>
    """

    # Nhận dữ liệu ảnh dán hoặc tải file
    col_paste, col_upload = st.columns(2)
    
    with col_paste:
        pasted_data = st.components.v1.html(paste_html, height=130)

    with col_upload:
        uploaded_file = st.file_uploader("Hoặc chọn tệp ảnh truyền thống:", type=["jpg", "jpeg", "png"])

    # Xử lý lấy ảnh vào PIL
    img = None
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.session_state["current_img"] = img

    # Nếu người dùng dán ảnh, Streamlit sẽ lưu qua session hoặc component
    if "current_img" in st.session_state and img is None:
        img = st.session_state["current_img"]

    if img and api_key:
        col_img, col_res = st.columns([1, 2])
        with col_img:
            st.image(img, caption="Ảnh chữ Hán cần phân tích", use_container_width=True)

        if st.button("🚀 Bắt Đầu Phân Tích", type="primary"):
            with st.spinner("AI đang quét chữ và bóc tách bộ thủ..."):
                try:
                    client = genai.Client(api_key=api_key)
                    prompt = """
                    Quét toàn bộ chữ Hán có trong ảnh này:
                    1. Nhận diện chính xác từng chữ Hán.
                    2. Phân tích thành phần bộ thủ thực tế.
                    3. Gợi ý mẹo nhớ (Mnemonics) ngắn gọn bằng tiếng Việt, kết nối bộ thủ với ý nghĩa chữ.
                    Trả về đúng định dạng schema được yêu cầu.
                    """
                    response = client.models.generate_content(
                        model="gemini-1.5-flash",
                        contents=[img, prompt],
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=OCRAnalysisResult,
                            temperature=0.2
                        )
                    )
                    data = OCRAnalysisResult.model_validate_json(response.text)
                    st.session_state["scanned_items"] = [item.model_dump() for item in data.words]
                except Exception as e:
                    st.error(f"Lỗi: {e}")

        if "scanned_items" in st.session_state:
            items = st.session_state["scanned_items"]
            with col_res:
                st.subheader(f"🔍 Nhận diện được {len(items)} chữ")
                if st.button("💾 Lưu tất cả vào sổ tay", type="secondary"):
                    added = save_new_words(items)
                    st.success(f"Đã lưu thành công {added} chữ mới vào sổ tay!")

                for item in items:
                    with st.expander(f"**{item['character']}** [{item['pinyin']}] — {item['sino_vietnamese']}: {item['meaning']}", expanded=True):
                        st.markdown(f"**Cấu tạo / Bộ:** {', '.join(item['components'])}")
                        st.info(f"💡 **Mẹo nhớ:** {item['mnemonics']}")
    elif not api_key:
        st.info("Vui lòng dán API Key vào cột bên trái để sử dụng.")

with tab_vault:
    df_saved = load_data()
    st.subheader(f"Tổng số từ đang lưu: {len(df_saved)}")

    if not df_saved.empty:
        search_kw = st.text_input("🔍 Tìm kiếm từ (gõ chữ Hán, Pinyin hoặc âm Hán Việt):").strip().lower()
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
            st.caption(f"Thời gian lưu: {row.get('created_at', 'Chưa rõ')}")
            st.divider()

        csv_data = df_saved.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 Tải toàn bộ sổ tay về máy (.CSV xem bằng Excel)",
            data=csv_data,
            file_name="so_tay_chu_han.csv",
            mime="text/csv"
        )
    else:
        st.write("Chưa có từ nào được lưu. Hãy quét ảnh bên tab kia và bấm lưu.")
