import streamlit as st
import numpy as np
import pandas as pd

# Đọc dữ liệu từ tệp .pkl
lr_model = pd.read_pickle('model/lr_model.pkl')
gbr_model = pd.read_pickle('model/gbr_model.pkl')

st.header("Dự đoán giá nhà")

# Tạo hai cột
left_column, right_column = st.columns(2)

# Ánh xạ các lựa chọn từ "Tốt" đến "Xấu" thành các giá trị từ 5 đến 1
condition_mapping = {5: "Tốt", 4: "Khá tốt", 3: "Bình thường", 2: "Khá xấu", 1: "Xấu"}

def predict_house_price(model, new_house):
    # Chuyển đổi mẫu dữ liệu mới thành định dạng phù hợp cho mô hình
    new_data_processed = np.array(new_house).reshape(1, -1)
    # Dự đoán giá nhà cho mẫu dữ liệu mới
    predicted_price = model.predict(new_data_processed)
    return predicted_price[0]

with left_column:
    bedrooms = st.number_input("Số phòng ngủ", min_value=0, max_value=33)
    bathrooms = st.number_input("Số phòng tắm/phòng ngủ", min_value=0.0, max_value=3.0, step=0.25)
    sqft_living = st.number_input("Diện tích căn nhà", min_value=500, max_value=4000)
    sqft_lot = st.number_input("Diện tích lô đất", min_value=1000, max_value=11000)
    sqft_basement = st.number_input("Diện tích tầng hầm", min_value=0, max_value=5000)
    floors = st.number_input("Số tầng", min_value=0.0, max_value=4.0, step=0.5)
    zipcode = st.number_input("Zip code", min_value=98000, max_value=98999, step=1)

with right_column:
    waterfront_option = st.selectbox("Có tầm nhìn bờ sông không", ("No", "Yes"))
    if waterfront_option == "No":
        waterfront = 0
    else:
        waterfront = 1
    view = st.number_input("Số mặt tiền", min_value=0, max_value=4, step=1)
    condition = st.selectbox("Tình trạng chung", (5, 4, 3, 2, 1), format_func=lambda x: condition_mapping[x])
    grade = st.number_input("Cấp độ", min_value=1, max_value=13, step=1)
    yr_built = st.number_input("Năm xây dựng", min_value=1900, max_value=2030)
    yr_renovated = st.number_input("Năm sửa chữa", min_value=0, max_value=2030, value=0)

predict = st.button("Dự đoán")

if predict:
    new_house = [bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, view, condition, grade, sqft_basement, yr_built, yr_renovated, zipcode]
    predicted_price = predict_house_price(gbr_model, new_house)
    formatted_price = f"{round(predicted_price):,} $".replace(',', '.')
    st.markdown(f"<h2 style='color: green; font-size: 25px'>Giá nhà được dự đoán là: {formatted_price}</h2>", unsafe_allow_html=True)
