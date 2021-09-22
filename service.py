import torch

from data import preprocess
from model import get_model
from visualize import view_classify
import streamlit as st


device = torch.device ('cuda' if torch.cuda.is_available () else 'cpu')
model = get_model ('weights/MobileNetV2.pth', device)



st.title ('Pathology Prediction')
st.write ('**Chest X-Ray**')

file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

if file is None:
    st.text("Please upload an image file:")
else:
    img = preprocess (file)
    
    option = st.sidebar.selectbox ('Select Your Visualize mode:',
                                  ['Figure', 'Table'])
    
    if option == 'Figure':
        st.write (view_classify (img, model, device) [0])
    else:
        st.write (view_classify (img, model, device) [1])
        
    st.write('You selected:', option)