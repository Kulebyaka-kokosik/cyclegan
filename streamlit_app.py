import streamlit as st
import pandas as pd
import math
from pathlib import Path

st.title("Domains transfer")
uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
