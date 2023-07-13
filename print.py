"""https://www.freakyjolly.com/multipage-canvas-pdf-using-jspdf/"""
import requests
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components


@st.cache_data()
def load_unpkg(src: str) -> str:
    return requests.get(src).text


HTML_2_CANVAS = load_unpkg("https://unpkg.com/html2canvas@1.4.1/dist/html2canvas.js")
JSPDF = load_unpkg("https://unpkg.com/jspdf@latest/dist/jspdf.umd.min.js")
BUTTON_TEXT = "Create PDF"

df = px.data.iris()

st.title("Download page as PDF")

with st.container():
    st.header("Big one")
    st.markdown(
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
    )
    st.plotly_chart(px.scatter(df, x="sepal_width", y="sepal_length", color="species"))
with st.container():
    st.header("Big 2")
    st.markdown(
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
    )
    st.plotly_chart(px.scatter(df, x="sepal_width", y="sepal_length", color="species"))
with st.container():
    st.header("Big 3")
    st.markdown(
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
    )
    st.plotly_chart(px.scatter(df, x="sepal_width", y="sepal_length", color="species"))
with st.container():
    st.header("Big 4")
    st.markdown(
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
    )
    st.plotly_chart(px.scatter(df, x="sepal_width", y="sepal_length", color="species"))

with st.sidebar:
    st.header("Configuration")
    st.slider("Slide me to reset ability to pdf", 0, 100)
    if st.button(BUTTON_TEXT):
        components.html(
            f"""
<script>{HTML_2_CANVAS}</script>
<script>{JSPDF}</script>
<script>
const html2canvas = window.html2canvas
const {{ jsPDF }} = window.jspdf
const streamlitDoc = window.parent.document;
const stApp = streamlitDoc.querySelector('.main > .block-container');
const buttons = Array.from(streamlitDoc.querySelectorAll('.stButton > button'));
const pdfButton = buttons.find(el => el.innerText === '{BUTTON_TEXT}');
const docHeight = stApp.scrollHeight;
const docWidth = stApp.scrollWidth;
console.log(stApp)
console.log(docHeight)
console.log(docWidth)
let topLeftMargin = 15;
let pdfWidth = docHeight + (topLeftMargin * 2);
let pdfHeight = (pdfWidth * 1.5) + (topLeftMargin * 2);
let canvasImageWidth = docWidth;
let canvasImageHeight = docHeight;
let totalPDFPages = Math.ceil(docHeight / pdfHeight)-1;
pdfButton.innerText = 'Creating PDF...';
html2canvas(stApp, {{ allowTaint: true }}).then(function (canvas) {{
    canvas.getContext('2d');
    let imgData = canvas.toDataURL("image/jpeg", 1.0);
    let pdf = new jsPDF('p', 'px', [pdfWidth, pdfHeight]);
    pdf.addImage(imgData, 'JPG', topLeftMargin, topLeftMargin, canvasImageWidth, canvasImageHeight);
    for (var i = 1; i <= totalPDFPages; i++) {{
        pdf.addPage();
        pdf.addImage(imgData, 'JPG', topLeftMargin, -(pdfHeight * i) + (topLeftMargin*4), canvasImageWidth, canvasImageHeight);
    }}
    pdf.save('test.pdf');
    pdfButton.innerText = '{BUTTON_TEXT}';
}})
</script>
""",
            height=0,
            width=0,
        )