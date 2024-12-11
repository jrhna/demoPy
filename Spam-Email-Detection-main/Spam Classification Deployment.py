import streamlit as st
import pickle
from PIL import Image
st.set_page_config(page_title="Spam E-Mail Classification")

# Use color and font themes
st.markdown("""
<style>

div[class*="stTextInput"] label p {
  font-size: 26px;
}
</style>
""", unsafe_allow_html=True)

tfidf = pickle.load(open(r"E:/RHNA/Visual/3_Py/Spam mail/Spam-Email-Detection-main/Spam-Email-Detection-main/Pickle Files/feature_new.pkl", 'rb'))
model = pickle.load(open(r"E:/RHNA/Visual/3_Py/Spam mail/Spam-Email-Detection-main/Spam-Email-Detection-main/Pickle Files/model_new.pkl", 'rb'))

st.title("Spam E-Mail Classifier")

image = Image.open(r"E:\RHNA\Visual\3_Py\Spam mail\Spam-Email-Detection-main\Spam-Email-Detection-main\Data Source\images.jpg")
st.image(image, use_column_width=True)


input_mail = st.text_input("Enter the Message")

if st.button('Predict'):

    vector_input = tfidf.transform([input_mail])

    result = model.predict(vector_input)

    st.success("This is a " + ('Spam Mail' if result == 0 else 'Ham Mail'))