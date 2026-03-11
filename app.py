import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re

# --- PHẦN 1: TÁI HIỆN LOGIC XỬ LÝ TỪ NOTEBOOK ---

def clean_vntitle(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[:;,.!?]', ' ', text)
    # Từ điển sửa lỗi từ notebook
    corrections = {
        r'\bko\b': 'không', r'\bk\b': 'không', r'\bdc\b': 'được', 
        r'\bwarning\b': 'cảnh_báo', r'\battendance\b': 'chuyên_cần'
    }
    for pattern, replacement in corrections.items():
        text = re.sub(pattern, replacement, text)
    text = re.sub(r'[^\w\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def apply_full_preprocessing(input_df, model_features):
    df = input_df.copy()
    
    # 1. Xử lý Chuyên cần (Attendance)
    att_cols = [c for c in model_features if 'Att_Subject' in c]
    for col in att_cols:
        if col not in df.columns:
            df[col] = 16 # Mặc định các môn khác là đi học đầy đủ
        df[col] = df[col].fillna(-1).clip(-1, 16)
    
    df['Actual_Subj_Count'] = (df[att_cols] > -1).sum(axis=1)
    temp_att = df[att_cols].replace(-1, np.nan)
    df['Total_Absence'] = (16 - temp_att).sum(axis=1)
    df['Avg_Absence_Per_Subj'] = df['Total_Absence'] / (df['Actual_Subj_Count'] + 1)
    df['Avg_Att_Rate'] = temp_att.mean(axis=1) / 16
    df['Att_Volatility'] = temp_att.std(axis=1).fillna(0)
    
    # 2. Xử lý Text & Stats
    df['Advisor_Notes_Clean'] = df['Advisor_Notes'].apply(clean_vntitle)
    df['Personal_Essay_Clean'] = df['Personal_Essay'].apply(clean_vntitle)
    
    neg_adv = ['không đến lớp', 'lười biếng', 'cảnh báo', 'bỏ tiết', 'yếu']
    pos_adv = ['tốt', 'chăm', 'nỗ lực', 'cố gắng', 'tiến bộ']
    
    df['Note_Neg_Score'] = df['Advisor_Notes_Clean'].apply(lambda x: sum(1 for w in neg_adv if w in x))
    df['Note_Pos_Score'] = df['Advisor_Notes_Clean'].apply(lambda x: sum(1 for w in pos_adv if w in x))
    df['Essay_Neg_Score'] = df['Personal_Essay_Clean'].apply(lambda x: sum(1 for w in neg_adv if w in x))
    df['Essay_Pos_Score'] = df['Personal_Essay_Clean'].apply(lambda x: sum(1 for w in pos_adv if w in x))
    
    # Công thức Advisor Trust Score từ notebook
    # $$Score = (Note_{Pos} \times 3) - (Note_{Neg} \times 5)$$
    df['Advisor_Trust_Score'] = (df['Note_Pos_Score'] * 3) - (df['Note_Neg_Score'] * 5)
    df['Conflict_Flag'] = ((df['Essay_Pos_Score'] > 2) & (df['Note_Neg_Score'] > 0)).astype(int)
    df['Advisor_Red_Flag'] = df['Note_Neg_Score'].apply(lambda x: 1 if x > 0 else 0)
    df['Notes_Len'] = df['Advisor_Notes_Clean'].str.len()
    df['Essay_Len'] = df['Personal_Essay_Clean'].str.len()
    
    # 3. Đặc trưng khác
    df['City_From_Address'] = df['Current_Address'].apply(lambda x: x.split(',')[-1].strip().lower() if ',' in x else 'unknown')
    df['Is_Away_From_Home'] = (df['Hometown'] != df['City_From_Address']).astype(int)
    df['Is_Mature_Student'] = (df['Age'] > 22).astype(int)
    df['Is_In_Debt'] = (df['Tuition_Debt'] > 0).astype(int)
    df['Financial_Stress_Index'] = df['Tuition_Debt'] / (df['Training_Score_Mixed'] + 1)
    df['Academic_Financial_Stress'] = df['Count_F'] * df['Is_In_Debt']
    
    # Đảm bảo thứ tự cột phải giống hệt lúc train
    return df[model_features]

# --- PHẦN 2: GIAO DIỆN APP ---

st.set_page_config(page_title="BAV Academic Warning", layout="wide")
st.title("Hệ thống Dự đoán Cảnh báo Học vụ BAV")

@st.cache_resource
def get_model():
    with open('model_v1.pkl', 'rb') as f:
        return pickle.load(f)

model = get_model()
# Lấy danh sách feature model yêu cầu
model_features = model.feature_names_

with st.sidebar:
    st.header("Thông tin Sinh viên")
    age = st.slider("Tuổi", 18, 50, 20)
    gender = st.selectbox("Giới tính", ["nam", "nữ"])
    hometown = st.text_input("Quê quán (Tỉnh)", "hà nội").lower()
    address = st.text_input("Địa chỉ hiện tại", "Chùa Bộc, Hà Nội")
    admission = st.selectbox("Hình thức nhập học", ["học bạ", "thi thpt", "tuyển thẳng"])

col1, col2 = st.columns(2)
with col1:
    st.subheader("Học tập & Tài chính")
    debt = st.number_input("Nợ học phí (VNĐ)", min_value=0, value=0)
    score = st.slider("Điểm rèn luyện", 0, 100, 75)
    count_f = st.number_input("Số môn bị điểm F", 0, 10, 0)
    eng = st.selectbox("Trình độ Tiếng Anh", ["ielts_high", "ielts_low", "unknown"])

with col2:
    st.subheader("Chuyên cần (Demo 3 môn)")
    att1 = st.slider("Môn 1 (Số buổi đi học)", 0, 16, 16)
    att2 = st.slider("Môn 2 (Số buổi đi học)", 0, 16, 16)
    att3 = st.slider("Môn 3 (Số buổi đi học)", 0, 16, 16)

st.subheader("Phần tự thuật & Đánh giá")
notes = st.text_area("Ghi chú của cố vấn học tập", "Sinh viên đi học đầy đủ.")
essay = st.text_area("Bài luận cá nhân", "Tôi hứa sẽ cố gắng.")

if st.button("Chạy dự đoán"):
    # Tạo dataframe thô
    raw_data = pd.DataFrame([{
        'Age': age, 'Gender': gender, 'Hometown': hometown,
        'Current_Address': address, 'Tuition_Debt': debt,
        'Training_Score_Mixed': score, 'Count_F': count_f,
        'English_Level': eng, 'Admission_Mode': admission,
        'Advisor_Notes': notes, 'Personal_Essay': essay,
        'Club_Member': 'unknown',
        'Att_Subject_1': att1, 'Att_Subject_2': att2, 'Att_Subject_3': att3
    }])
    
    # Tiền xử lý để khớp với 65 cột của model
    final_input = apply_full_preprocessing(raw_data, model_features)
    
    # Dự đoán
    # Sử dụng .item() hoặc [0] thêm lần nữa để lấy đúng giá trị số bên trong
    prediction_raw = model.predict(final_input)
    pred = int(prediction_raw.flatten()[0]) 

    # Sau đó dòng hiển thị sẽ chạy mượt mà
    st.markdown(f"### Kết quả: :{colors[pred]}[{labels[pred]}]")
    prob = model.predict_proba(final_input)[0]
    
    # Hiển thị kết quả
    labels = {0: "Bình thường", 1: "Cảnh báo", 2: "Nguy cơ thôi học"}
    colors = {0: "green", 1: "orange", 2: "red"}
    
    st.markdown(f"### Kết quả: :{colors[int(pred)]}[{labels[int(pred)]}]")
    st.write(f"Độ tin cậy: {np.max(prob)*100:.2f}%")