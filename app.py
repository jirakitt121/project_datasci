import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. ตั้งค่าหน้าเว็บ (แท็บเบราว์เซอร์)
st.set_page_config(page_title="AI Price Predictor", page_icon="🛍️", layout="centered")

# 2. ฟังก์ชันโหลดโมเดล (อ่านไฟล์ model_light.joblib ตรงๆ เลย)
@st.cache_resource
def load_model():
    try:
        # ✅ แก้ชื่อไฟล์ให้ตรงกับที่คุณโหลดมาเป๊ะๆ แล้วครับ
        return joblib.load('model_light.joblib')
    except Exception as e:
        return None

model = load_model()

# 3. ส่วนหัวของเว็บ
st.title("🛍️ AI ประเมินราคาสินค้ามือสอง")
st.markdown("กรอกข้อมูลสินค้าของคุณด้านล่าง แล้วให้ AI (Ridge + TF-IDF) ช่วยประเมินราคาที่เหมาะสมให้ครับ!")
st.divider()

# 4. จัด Layout ฟอร์มรับข้อมูลให้ดูสวยงาม
with st.form("prediction_form"):
    st.subheader("📦 ข้อมูลสินค้า")
    
    name = st.text_input("ชื่อสินค้า (Name) *", placeholder="เช่น Women's Levi's shorts")
    
    col1, col2 = st.columns(2)
    with col1:
        item_condition_id = st.selectbox("สภาพสินค้า (1=ใหม่มาก, 5=แย่)", [1, 2, 3, 4, 5])
        brand_name = st.text_input("แบรนด์ (Brand)", placeholder="เช่น Levi's, Apple")
        
    with col2:
        shipping_text = st.selectbox("การจัดส่ง", ["ผู้ซื้อจ่าย (Buyer Pays)", "ผู้ขายจ่าย (Free Shipping)"])
        category_name = st.text_input("หมวดหมู่ (Category)", placeholder="เช่น Women/Athletic Apparel/Shorts")

    item_description = st.text_area("รายละเอียดสินค้า (Description)", placeholder="อธิบายสภาพ ขนาด หรือตำหนิของสินค้า...")

    submit_button = st.form_submit_button("✨ วิเคราะห์และคำนวณราคา", use_container_width=True)

# 5. การทำงานเมื่อผู้ใช้กดปุ่ม
if submit_button:
    if model is None:
        # ✅ แก้ข้อความแจ้งเตือนให้ตรงกัน
        st.error("❌ ไม่พบไฟล์โมเดล 'model_light.joblib' กรุณาตรวจสอบใน GitHub ของคุณครับ")
    else:
        with st.spinner("⏳ AI กำลังประมวลผล..."):
            
            shipping_val = 1 if shipping_text == "ผู้ขายจ่าย (Free Shipping)" else 0
            
            input_df = pd.DataFrame({
                'name': [name],
                'item_condition_id': [item_condition_id],
                'category_name': [category_name],
                'brand_name': [brand_name],
                'shipping': [shipping_val],
                'item_description': [item_description]
            })

            # --- Data Cleaning ---
            input_df.replace("", np.nan, inplace=True)
            
            input_df['category_name'] = input_df['category_name'].fillna('Other/Other/Other')
            input_df['item_description'] = input_df['item_description'].fillna('No description')
            input_df['brand_name'] = input_df['brand_name'].fillna('Unknown')
            input_df['name'] = input_df['name'].fillna('Unknown')

            input_df['cat1'] = input_df['category_name'].apply(lambda x: str(x).split('/')[0] if '/' in str(x) else 'Other')
            input_df['cat2'] = input_df['category_name'].apply(lambda x: str(x).split('/')[1] if str(x).count('/') >= 1 else 'Other')
            input_df['cat3'] = input_df['category_name'].apply(lambda x: str(x).split('/')[2] if str(x).count('/') >= 2 else 'Other')

            # --- Predict ---
            try:
                prediction_log = model.predict(input_df)
                predicted_price = np.expm1(prediction_log[0])
                
                if predicted_price < 0:
                    predicted_price = 0

                st.success("🎉 ประมวลผลเสร็จสิ้น!")
                st.metric(label="💰 ราคาที่ AI แนะนำ (USD)", value=f"${predicted_price:,.2f}")
                st.balloons()
                
            except Exception as e:
                st.error(f"❌ เกิดข้อผิดพลาดในการคำนวณ: {e}")