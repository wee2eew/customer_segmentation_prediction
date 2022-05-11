import pandas as pd

from sklearn.cluster import KMeans

def build_model(df):
    # เลือกเฉพาะ feature 'Age', 'Annual Income (k$)', 'Spending Score (1-100)'' ดูก่อน
    X1 = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].copy()

    # ใช้ฟังก์ชัน KMeans โดยกำหนดพารามิเตอร์ดังนี้
    # n_clusters คือ จำนวน k กลุ่ม
    # max_iter คือ จำนวนรอบสูงสุดของการวนเพื่อคำนวนค่า centroid ใหม่ของกลุ่ม
    # random_state คือ ค่าตัวเลข seed การสุ่มเริ่มต้น
    Model1 = (KMeans(n_clusters = 3 ,max_iter=300,  random_state= 111))

    # ใช้คำสั่ง .fit() เพื่อเทรนโมเดล
    Model1.fit(X1)

    return Model1


def get_user_test_data(age, annual_income, spending_score):
    user = [age, annual_income, spending_score]
    X_test = pd.DataFrame([user], columns =[ 'Age', 'Annual Income (k$)', 'Spending Score (1-100)'])
    print(X_test)
    return X_test

def pipeline(age, annual_income, spending_score):
    # get data
    df = pd.read_csv("mall_200customers.csv", sep=",")

    # build model
    Model1 = build_model(df)

    # get user_test_data
    X_test = get_user_test_data(age, annual_income, spending_score)

    # predict new user test data from our model
    y_pred = Model1.predict(X_test)

    if (y_pred == 0):
        return "คุณเป็นกลุ่มลูกค้าอายุน้อย แต่มีกำลังซื้อเยอะ"
    elif (y_pred == 1):
        return "คุณเป็นกลุ่มที่มีอายุหลากหลาย แต่มีกำลังซื้อไม่มาก"
    else:
        return "คุณเป็นกลุ่มที่มีอายุหลากหลาย แต่มีกำลังปานกลาง"
