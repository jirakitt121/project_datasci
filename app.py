import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="AI Price Predictor", page_icon="🛍️", layout="centered")

# 🔥 โหลดไฟล์ชื่อใหม่ เพื่อหนีระบบจำ (Cache) ของ Streamlit
@st.cache_resource
def load_model():
    try:
        return joblib.load('model_v2.joblib')
    except Exception as e:
        return None

model = load_model()

st.title("🛍️ AI ประเมินราคาสินค้ามือสอง")
st.markdown("กรอกข้อมูลสินค้าของคุณด้านล่าง แล้วให้ AI (Ridge + TF-IDF) ช่วยประเมินราคาที่เหมาะสมให้ครับ!")
st.divider()

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
    
    # 🔥 ส่วนที่เพิ่มใหม่: ช่องรับราคาสมมุติ
    user_price = st.number_input("💵 ราคาที่คุณต้องการขาย (USD) - ใส่ 0 ถ้ายากให้ AI แนะนำอย่างเดียว", min_value=0.0, step=1.0)
    
    submit_button = st.form_submit_button("✨ วิเคราะห์และคำนวณราคา", use_container_width=True)

if submit_button:
    if model is None:
        st.error("❌ ไม่พบไฟล์โมเดล 'model_v2.joblib' กรุณาตรวจสอบใน GitHub ของคุณครับ")
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

            input_df.replace("", np.nan, inplace=True)
            input_df['category_name'] = input_df['category_name'].fillna('Other/Other/Other')
            input_df['item_description'] = input_df['item_description'].fillna('No description')
            input_df['brand_name'] = input_df['brand_name'].fillna('Unknown')
            input_df['name'] = input_df['name'].fillna('Unknown')

            input_df['cat1'] = input_df['category_name'].apply(lambda x: str(x).split('/')[0] if '/' in str(x) else 'Other')
            input_df['cat2'] = input_df['category_name'].apply(lambda x: str(x).split('/')[1] if str(x).count('/') >= 1 else 'Other')
            input_df['cat3'] = input_df['category_name'].apply(lambda x: str(x).split('/')[2] if str(x).count('/') >= 2 else 'Other')

            cols_to_str = ['item_condition_id', 'shipping', 'brand_name', 'cat1', 'cat2', 'cat3']
            for c in cols_to_str:
                input_df[c] = input_df[c].astype(str)

            # 🔥 บังคับเลือก 8 คอลัมน์ (ใส่ไว้กัน Error 2005 features)
            expected_features = ['name', 'item_condition_id', 'brand_name', 'shipping', 'item_description', 'cat1', 'cat2', 'cat3']
            final_df = input_df[expected_features]

            try:
                prediction_log = model.predict(final_df)
                predicted_price = np.expm1(prediction_log[0])
                if predicted_price < 0:
                    predicted_price = 0

                st.success("🎉 ประมวลผลเสร็จสิ้น!")
                
                # โชว์ราคา AI
                st.metric(label="💰 ราคาที่ AI แนะนำ (USD)", value=f"${predicted_price:,.2f}")
                
                # 🔥 ส่วนที่เพิ่มใหม่: ระบบเปรียบเทียบราคา
                if user_price > 0:
                    diff = user_price - predicted_price
                    diff_pct = (diff / predicted_price) * 100 if predicted_price > 0 else 0
                    
                    if diff > (predicted_price * 0.15): # ตั้งราคาสูงกว่า AI เกิน 15%
                        st.warning(f"⚠️ **วิเคราะห์ราคา:** คุณตั้งราคาสูงกว่าตลาดประมาณ {diff_pct:.1f}% สินค้าอาจจะขายออกได้ช้าครับ")
                    elif diff < -(predicted_price * 0.15): # ตั้งราคาต่ำกว่า AI เกิน 15%
                        st.info(f"✨ **วิเคราะห์ราคา:** คุณตั้งราคาต่ำกว่าตลาดประมาณ {abs(diff_pct):.1f}% น่าจะขายออกไวมากแน่นอน!")
                    else: # ราคาใกล้เคียงตลาด
                        st.success(f"✅ **วิเคราะห์ราคา:** ราคาที่คุณตั้งเหมาะสมและแข่งขันได้ดีมากครับ!")

                st.balloons()
            except Exception as e:
                st.error(f"❌ เกิดข้อผิดพลาดในการคำนวณ: {e}")