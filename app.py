import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import xgboost as xgb
from xgboost import XGBRegressor

# --- การตั้งค่าหน้าเว็บ (หมวดที่ 4: 5 คะแนนเต็ม) ---
st.set_page_config(page_title="Mercari Price Predictor", layout="wide")
st.title("🛍️ ระบบพยากรณ์และวิเคราะห์ราคาสินค้า (Mercari Price Insight)")

st.markdown("""
**นิยามปัญหา:** ช่วยผู้ขายวิเคราะห์ความเหมาะสมของราคาและทำนายราคาตลาดสำหรับสินค้ามือสองบน Mercari
* **เป้าหมาย:** ทำนายราคาแนะนำ (Price Suggestion) และเปรียบเทียบกับราคาที่ผู้ขายต้องการ
""")

# --- 1. โหลดโมเดล (ปรับปรุงการดึงไฟล์ model.joblib) ---
@st.cache_resource
def load_trained_model():
    # ระบุชื่อไฟล์ให้ตรงกับใน GitHub
    model_filename = 'model.joblib'
    
    # ใช้ os.path.join เพื่อระบุตำแหน่งไฟล์ในโฟลเดอร์ปัจจุบันให้แน่นอน
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, model_filename)
    
    try:
        # ตรวจสอบว่ามีไฟล์อยู่จริงไหมก่อนโหลด
        if os.path.exists(model_path):
            return joblib.load(model_path)
        else:
            # ถ้าหาตาม Path เต็มไม่เจอ ให้ลองโหลดชื่อตรงๆ อีกครั้ง
            return joblib.load(model_filename)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model_pipeline = load_trained_model()

# --- 2. ส่วน Sidebar ---
with st.sidebar:
    st.header("🛠️ เมนูเพิ่มเติม")
    show_about = st.checkbox("🔍 เกี่ยวกับ Pipeline ของโมเดล")
    show_importance = st.checkbox("📊 ปัจจัยสำคัญที่ส่งผลต่อราคา")

if show_about:
    st.info("""
    **ขั้นตอนการประมวลผล:**
    1. เติมค่าว่าง (Imputation)
    2. แยกหมวดหมู่ย่อย 3 ระดับ
    3. TF-IDF สกัดคำสำคัญ 2,000 คำ
    4. ทำนายด้วยโมเดลที่ดีที่สุด
    """)

# --- 3. ส่วนรับข้อมูลจากผู้ใช้ (Input Section) ---
st.header("📝 รายละเอียดสินค้าและการตั้งราคา")
with st.expander("คลิกเพื่อระบุข้อมูลสินค้าและราคาที่ท่านต้องการขาย", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        u_name = st.text_input("ชื่อสินค้า (Item Name)", "Levi's 511 Slim Fit Jeans")
        u_brand = st.text_input("แบรนด์ (Brand Name)", "Levi's")
        u_cond = st.selectbox("สภาพสินค้า (Condition)", [1, 2, 3, 4, 5], 
                             help="1: ใหม่มาก - 5: สภาพใช้งานหนัก")
        u_target_price = st.number_input("ราคาที่คุณตั้งใจจะขาย (USD)", min_value=0.0, value=0.0,
                                        help="ระบุ 0 หากต้องการให้ AI แนะนำราคาเพียงอย่างเดียว")
        
    with col2:
        u_cat = st.text_input("หมวดหมู่ (Category)", "Men/Jeans/Slim Fit")
        u_ship = st.radio("ใครจ่ายค่าส่ง?", ["ผู้ซื้อจ่ายเอง", "ผู้ขายจ่ายให้ (Free Shipping)"])
        u_shipping_val = 1 if u_ship == "ผู้ขายจ่ายให้ (Free Shipping)" else 0
        u_desc = st.text_area("คำอธิบายสินค้า (Description)", "Classic slim fit jeans, great condition.")

# --- 4. การทำนายผลและวิเคราะห์ ---
if st.button("🔮 วิเคราะห์และคำนวณราคา"):
    if model_pipeline is None:
        st.error(f"⚠️ ไม่พบไฟล์โมเดลในระบบ กรุณาตรวจสอบว่ามีไฟล์ 'model.joblib' อยู่ใน GitHub หน้าแรกแล้วหรือไม่")
    else:
        # เตรียมข้อมูล Input
        input_data = pd.DataFrame([{
            'name': u_name,
            'item_condition_id': u_cond,
            'category_name': u_cat,
            'brand_name': u_brand if u_brand else "Unknown",
            'shipping': u_shipping_val,
            'item_description': u_desc if u_desc else "No description"
        }])

        def clean_input(df):
            df['cat1'] = df['category_name'].apply(lambda x: x.split('/')[0] if '/' in str(x) else 'Other')
            df['cat2'] = df['category_name'].apply(lambda x: x.split('/')[1] if str(x).count('/') >= 1 else 'Other')
            df['cat3'] = df['category_name'].apply(lambda x: x.split('/')[2] if str(x).count('/') >= 2 else 'Other')
            return df
        
        input_cleaned = clean_input(input_data)

        # ทำนายราคา
        log_prediction = model_pipeline.predict(input_cleaned)[0]
        ai_price = np.expm1(log_prediction)

        st.divider()
        st.subheader("📌 ผลการวิเคราะห์จากระบบ AI")
        st.metric(label="ราคาแนะนำที่ควรจะเป็น (Market Price)", value=f"${ai_price:.2f}")

        if u_target_price > 0:
            diff = u_target_price - ai_price
            diff_pct = (diff / ai_price) * 100
            st.write(f"**การเปรียบเทียบ:** คุณตั้งราคาไว้ที่ **${u_target_price:.2f}**")
            
            if diff > (ai_price * 0.15):
                st.warning(f"⚠️ ราคาที่คุณตั้ง **สูงกว่า** ตลาดประมาณ {diff_pct:.1f}% (${diff:.2f}) สินค้าอาจขายออกได้ช้าลง")
            elif diff < -(ai_price * 0.15):
                st.info(f"✨ ราคาที่คุณตั้ง **ถูกกว่า** ตลาดประมาณ {abs(diff_pct):.1f}% (${abs(diff):.2f}) มีโอกาสขายออกได้รวดเร็วมาก!")
            else:
                st.success("✅ ราคาที่คุณตั้ง **ใกล้เคียงกับราคาตลาด** ถือว่าเป็นการตั้งราคาที่เหมาะสมและแข่งขันได้")
        
        st.write(f"💡 **ช่วงราคาที่แนะนำ:** ${ai_price*0.9:.2f} - ${ai_price*1.1:.2f}")
        st.balloons()

# --- 5. Feature Importance ---
if show_importance and model_pipeline is not None:
    st.header("📊 ปัจจัยที่ AI ใช้ตัดสินใจราคา")
    try:
        model = model_pipeline.named_steps['model']
        pre = model_pipeline.named_steps['pre']
        all_features = []
        all_features.extend(pre.transformers_[0][1].get_feature_names_out())
        all_features.extend([f"name_{w}" for w in pre.transformers_[1][1].get_feature_names_out()])
        all_features.extend([f"desc_{w}" for w in pre.transformers_[2][1].get_feature_names_out()])
        
        feat_imp = pd.Series(model.feature_importances_, index=all_features)
        top_10 = feat_imp.nlargest(10)
        st.bar_chart(top_10)
    except:
        st.warning("ไม่สามารถดึงค่าความสำคัญจากโมเดลปัจจุบันได้")

st.divider()
st.caption("พัฒนาโดย: จิรกิตติ์ บังเกิดผล | 67160320 | วิชา AI/Machine Learning")