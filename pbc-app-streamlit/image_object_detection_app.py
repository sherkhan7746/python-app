import PIL
import streamlit as st
from ultralytics import YOLO


model_path = 'best.pt'

# Setting page layout
st.set_page_config(
    page_title="Object Detection using YOLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Creating sidebar
with st.sidebar:
    st.header("Image")
    source_img = st.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))
    confidence = float(st.slider(
        "Select Model Confidence", 25, 100, 40)) / 100

# Creating main page heading
st.title("Object Detection using YOLOv8")

col1, col2 = st.columns(2)

desired_width = 400

with col1:
    if source_img:
        uploaded_image = PIL.Image.open(source_img)
        # Display the uploaded image with the specified dimensions
        st.image(uploaded_image,
                 caption="Uploaded Image",
                 use_column_width=False,  # Ensure custom width and height are used
                 width=desired_width
                 )

try:
    model = YOLO("best.pt")
except Exception as ex:
    st.error(
        f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

if st.sidebar.button('Detect Objects'):
    res = model.predict(uploaded_image, conf=confidence)
    boxes = res[0].boxes
    res_plotted = res[0].plot()[:, :, ::-1]
    with col2:
        st.image(res_plotted,
                 caption='Detected Image',
                 use_column_width=False,
                 width=desired_width)
        try:
            with st.expander("Detection Results"):
                for box in boxes:
                    st.write(box.xywh)
        except Exception as ex:
            st.write("No image is uploaded yet!")
