# Import the basic libraries
import streamlit as st 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

# We will be creating an app with the library wordcloud, if you have not installed it already than
# I do recommend you to install it along with PyPDF2, docx and plotly.

# ----------------------------------------
#----------- INSTALLATIONS -----------
# !pip install wordcloud
# !pip install PyPDF2
# !pip install docx
# !pip install plotly
# ----------------------------------------

from wordcloud import WordCloud, STOPWORDS
import PyPDF2
import docx
import plotly.express as px

import base64
from io import BytesIO

# -------------------------------------------------------------------------

# Creating functions for reading different file formats
def read_txt(file):
    return file.getvalue().decode("utf-8")

def read_docx(file):
    doc = docx.Document(file)
    return " ".join([para.text for para in doc.paragraphs])

def read_pdf(file):
    pdfReader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdfReader.pages:
        text += page.extract_text()
    return text
# -------------------------------------------------------------------------

# Creating a function to remove all of the stopwords
def filter_stopwords(text, additional_stopwords=[]):
    words = text.split()
    all_stopwords = STOPWORDS.union(set(additional_stopwords))
    filtered_words = [word for word in words if word.lower() not in all_stopwords]
    return " ".join(filtered_words)

# -------------------------------------------------------------------------

# Creating a function to create download link for plot
def get_image_download_link(buffered, format_):
    image_base64 = base64.b64encode(buffered.getvalue()).decode()
    return f'<a href="data:image/{format_};base64,{image_base64}" download="wordcloud.{format_}">DOWNLOAD THE PLOT AS {format_}</a>'

# -------------------------------------------------------------------------

# Creating a function to generate a download link for a DataFrame
def get_table_download_link(df, filename, file_label):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">{file_label}</a>'

# -------------------------------------------------------------------------

# BASIC TITLES FOR OUR APP
st.title("Creative Word Clouds 🌟✨")

st.subheader("`Application created by:` Aarish Khan and Asif Ali")
st.subheader("`Date:` 5th April 2024")
st.markdown("---")
# -------------------------------------------------------------------------

st.write("# **Using the Application?**")

st.write("Upload a PDF, DOCX, or a TXT file, and witness your words come alive in a vibrant, captivating visual display!")

uploaded_file = st.file_uploader(" ---> Choose any file format📁 <---", type=["txt", "pdf", "docx"])
st.set_option('deprecation.showPyplotGlobalUse', False)

# -------------------------------------------------------------------------

if uploaded_file:
    file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
    st.write(file_details)

    # Check the file type and read the file
    if uploaded_file.type == "text/plain":
        text = read_txt(uploaded_file)
    elif uploaded_file.type == "application/pdf":
        text = read_pdf(uploaded_file)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        text = read_docx(uploaded_file)
    else:
        st.error("File type not supported. Please upload a txt, pdf or docx file.")
        st.stop()

    # Generate word count table
    words = text.split()
    word_count = pd.DataFrame({'Word': words}).groupby('Word').size().reset_index(name='Count').sort_values('Count', ascending=False)
# -------------------------------------------------------------------------

     # Sidebar: Checkbox and Multiselect box for stopwords
    use_standard_stopwords = st.sidebar.checkbox("Use the Standard Stopwords", True)
    top_words = word_count['Word'].head(50).tolist()
    additional_stopwords = st.sidebar.multiselect("The Additional Stopwords:", sorted(top_words))

    if use_standard_stopwords:
        all_stopwords = STOPWORDS.union(set(additional_stopwords))
    else:
        all_stopwords = set(additional_stopwords)

    text = filter_stopwords(text, all_stopwords)

    
    if text:
        # Word Cloud dimensions
        width = st.sidebar.slider(" ---> Select the Word Cloud Width:", 400, 2000, 1200, 50)
        height = st.sidebar.slider("---> Select the Word Cloud Height:", 200, 2000, 800, 50)

        # Generate wordcloud
        fig, ax = plt.subplots(figsize=(width/100, height/100))  # Convert pixels to inches for figsize
        wordcloud_img = WordCloud(width=width, height=height, background_color='white', max_words=200, contour_width=3, contour_color='steelblue').generate(text)
        ax.imshow(wordcloud_img, interpolation='bilinear')
        ax.axis('off')


      # Save plot functionality
        format_ = st.selectbox(" ---> Select a File format to Save the plot:", ["png", "jpeg", "svg", "pdf"])
        resolution = st.slider("--> Select the Resolution:", 100, 500, 300, 50)
# -------------------------------------------------------------------------
        # Generate word count table
        st.subheader("Word Count Table - Shows how many times a Word was repeated")
        words = text.split()
        word_count = pd.DataFrame({'Word': words}).groupby('Word').size().reset_index(name='Count').sort_values('Count', ascending=False)
        st.write(word_count)
        
        st.subheader("Word Cloud Plot")
        
    st.pyplot(fig)
    if st.button(f"SAVE AS", {format_}):
        buffered = BytesIO()
        plt.savefig(buffered, format=format_, dpi=resolution)
        st.markdown(get_image_download_link(buffered, format_), unsafe_allow_html=True)
    
    # Provide download link for table
    if st.button('DOWNLOAD THE WORD COUNT-TABLE'):
        st.markdown(get_table_download_link(word_count, "word_count.csv", "Click Here to Download"), unsafe_allow_html=True)
