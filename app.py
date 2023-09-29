import streamlit as st
import fastai
from fastai.vision.all import *
import plotly.express as px
import pathlib
import platform
plt=platform.system()
if plt =='Linux': pathlib.WindowsPath=pathlib.PosixPath
st.title('Transportni klassifikatsiya qiluvchi model')
file=st.file_uploader('Rasm yuklash', type=['png', 'jpeg', 'gif', 'svg'])
if file:
    img= PILImage.create(file)
    model=load_learner('transport_model.pkl')

    pred, pred_id, probs=model.predict(img)

    st.success(f"bashorat: {pred}")
    st.info(f'Ehtimollik: {probs[pred_id]*100:.1f}%')

    st.image(file)
    #plotting  
    fig=px.bar(x=probs*100, y=model.dls.vocab)
    st.plotly_chart(fig)  
