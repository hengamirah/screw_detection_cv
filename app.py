import streamlit as st

# Set html page configuration
import time
import cv2
import numpy as np
import tempfile
import time
from collections import Counter
import json
import pandas as pd
from PIL import ImageColor
#from model_utils import get_system_stat
import random
from PIL import Image
from ultralytics import YOLO
import io 
import torch
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase



st.set_page_config(page_title="Amirah Streamlit App", layout="wide", initial_sidebar_state="auto")

def get_yolo(img, model, confidence, color_pick_list, class_labels, draw_thick):
                    
    current_no_class = []
    results = model.predict(img)
    #res_plot= results[0].plot()[:, :, ::-1]
    #boxes = results[0].boxes
                      
    for result in results:
        bboxs = result.boxes.xyxy
        conf = result.boxes.conf
        cls = result.boxes.cls
        
        for bbox, cnf, cs in zip(bboxs, conf, cls):
            xmin = int(bbox[0]) 
            ymin = int(bbox[1])
            xmax = int(bbox[2])
            ymax = int(bbox[3])
            #st.write(f"Box: {xmin}, {ymin}, {xmax}, {ymax}") - Box location
            if cnf > confidence:
                plot_one_box([xmin, ymin, xmax, ymax], img, label=class_labels[int(cs)],
                                color=color_pick_list[int(cs)], line_thickness=draw_thick)
                current_no_class.append([class_labels[int(cs)]])
    return img, current_no_class

# from ultralytics import YOLO
def color_picker_fn(classname, key):
    color_picke = st.sidebar.color_picker(f'{classname}:', '#ff0003', key=key)
    color_rgb_list = list(ImageColor.getcolor(str(color_picke), "RGB"))
    color = [color_rgb_list[0], color_rgb_list[1],color_rgb_list[2] ]
    return color


def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

class MyVideoTransformer(VideoTransformerBase):
    def __init__(self, conf, model):
        self.conf = conf
        self.model = model

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        processed_image = self._display_detected_frames(image)
        st.image(processed_image, caption='Detected Video', channels="BGR", use_column_width=True)

    def _display_detected_frames(self, image):
        orig_h, orig_w = image.shape[0:2]
        width = 720  # Set the desired width for processing

        # cv2.resize used in a forked thread may cause memory leaks
        input = np.asarray(Image.fromarray(image).resize((width, int(width * orig_h / orig_w))))

        if self.model is not None:
            # Perform object detection using YOLO model
            res = self.model.predict(input, conf=self.conf)

            # Plot the detected objects on the video frame
            res_plotted = res[0].plot()
            return res_plotted

        return input


def predict():        
    #p_time = 0


    # Hide main menu style
    menu_style_cfg = """<style>MainMenu {visibility: hidden;}</style>"""

    # Main title of streamlit application
    main_title_cfg = """<div><h1 style="color:#BF0000; text-align:center; font-size:40px; 
                                font-family: 'Archivo', sans-serif; margin-top:-70px;margin-bottom:20px;">
                    Computer Vision Detection App
                    </h1></div>"""

    # Subtitle of streamlit application
    sub_title_cfg = """<div><h4 style="color:#f7b307; text-align:center; 
                    font-family: 'Archivo', sans-serif; margin-top:-30px; margin-bottom:10px;">
                    Experience real-time object detection on your webcam, uploaded video or image! 🚀</h4>
                    </div>"""
    # Custom CSS for sidebar and main page

    # Append the custom HTML
    st.markdown(menu_style_cfg, unsafe_allow_html=True)
    st.markdown(main_title_cfg, unsafe_allow_html=True)
    st.markdown(sub_title_cfg, unsafe_allow_html=True)
            
    # Add ultralytics logo in sidebar
    with st.sidebar:
        logo = "msf_logo.png"
        st.image(logo, width=250)

    st.sidebar.title('User Configuration')
        # Choose the model
    model_type = st.sidebar.selectbox(
        'Choose YOLO Model', ('Select Model','Screw Detection', 'Upload Model')
    )

    cap = None

    if model_type == 'Upload Model':
        path_model_file = st.sidebar.text_input(
            f'path to {model_type} Model:',
            f'eg: dir/best.pt'
        )

    # YOLOv8 Model
    elif model_type == 'Screw Detection':
        path_model_file = "best.pt"
        

    elif model_type == 'Select Model':
        pass
    else:
        st.error('Model not found!!')

    st.subheader(f'{model_type} Predictions')
    col1, col2 = st.columns([2,2])
    org_frame = col1.empty()
    ann_frame = col2.empty()

    pred = False
    pred1 = False
    pred2 = False
    if model_type!= 'Select Model':
        
        load = st.sidebar.checkbox("Load Model",key = 'Load Model')

        if load:
            with st.spinner("Model is downloading..."):
                model = YOLO(path_model_file)
                st.sidebar.success(f'Model loaded successfully!')
            # Load Class names
            class_labels = model.names
            
            # Inference Mode
            options = st.sidebar.radio('Options:', ('Webcam', 'Image', 'Video', 'RTSP'), index=1, key='options')

            # Confidence
            confidence = st.sidebar.slider( 'Detection Confidence', min_value=0.0, max_value=1.0, value=0.25,key='confidence')

            # Draw thickness
            draw_thick = st.sidebar.slider('Draw Thickness:', min_value=1,max_value=20, value=2, key='draw_thick' )

            # Image
            if options == 'Image':
                     
                color_pick_list = []
                for i in range(len(class_labels)):
                    classname = class_labels[i]
                    color = color_picker_fn(classname, i)
                    color_pick_list.append(color)

                upload_img_file = st.sidebar.file_uploader('Upload Image', type=['jpg', 'jpeg', 'png'],key ='image_uploader')
                
                if upload_img_file is not None:
                    pred = st.button("Start")
                    byte_data = upload_img_file.read()
                    image = Image.open(upload_img_file)
                    img= np.array(image)
                    org_frame.image(upload_img_file,caption='Uploaded Image', channels="BGR", width=300)
                    #st.image(upload_img_file, caption='Uploaded Image', use_column_width=True)  # Display the uploaded image in Streamlit
                    # display frame
                    

                    if pred:
                        with st.spinner("Predicting..."):
                            img, current_no_class = get_yolo(img, model, confidence, color_pick_list, class_labels, draw_thick)    
                            ann_frame.image(img,caption='Predicted Image', channels="RGB",width=300)
                            #st.image(img, channels='BGR',use_column_width=True)
                            # Current number of classes
                            class_fq = dict(Counter(i for sub in current_no_class for i in set(sub)))
                            class_fq = json.dumps(class_fq, indent = 4)
                            class_fq = json.loads(class_fq)
                            df_fq = pd.DataFrame(class_fq.items(), columns=['Class', 'Quantity'])
                            
                            # Updating Inference results
                            with st.container():
                                st.markdown("<h2>Inference Statistics</h2>", unsafe_allow_html=True)
                                st.markdown("<h3>Detected objects in current Frame</h3>", unsafe_allow_html=True)
                                st.dataframe(df_fq, use_container_width=True)

            # Video
            elif options == 'Video':
                upload_video_file = st.sidebar.file_uploader( 'Upload Video', type=['mp4', 'avi', 'mkv'],key ='vid_uploader')
                if upload_video_file is not None:
                    
                    g = io.BytesIO(upload_video_file.read()) # BytesIO Object
                    vid_location = "ultralytics.mp4"
                    with open(vid_location, "wb") as out:  # Open temporary file as bytes
                        out.write(g.read())  # Read bytes into file
                    vid_file_name = "ultralytics.mp4"
                    pred1 = st.sidebar.button("Start")
                    #tfile = tempfile.NamedTemporaryFile(delete=False)
                    #tfile.write(upload_video_file.read())
            
                if pred1:
        
                    class_names = list(model.names.values())# Convert dictionary to list of class names
                    selected_classes = st.sidebar.multiselect("Classes", class_names, default=class_names[:3],key='select_class')
                        
                    with st.spinner("Predicting..."):
                        fps_display = st.sidebar.empty()  # Placeholder for FPS display
                        
                        
                        cap = cv2.VideoCapture(vid_file_name)
                        st.write(vid_file_name)
                        if not cap.isOpened():
                            st.error("Could not open Video.")
                        
                        stop_button = st.button("Stop")  # Button to stop the inference

                        while cap.isOpened():
                            success, frame = cap.read()
                            if not success:
                                st.warning("Failed to read frame from webcam. Please make sure the webcam is connected properly.")
                                break
                            prev_time = time.time()

                            # Multiselect box with class names and get indices of selected classes
                            selected_ind = [class_names.index(option) for option in selected_classes]
                            if not isinstance(selected_ind, list):  # Ensure selected_options is a list
                                selected_ind = list(selected_ind)

                            results = model.track(frame, conf=confidence, classes=selected_ind, persist=True)
                            annotated_frame = results[0].plot()  # Add annotations on frame
                            
                            # Calculate model FPS
                            curr_time = time.time()
                            fps = 1 / (curr_time - prev_time)
                            prev_time = curr_time
                            # display frame
                            org_frame.image(frame, channels="BGR")
                            ann_frame.image(annotated_frame, channels="BGR")
                                
                            if stop_button:
                                cap.release()  # Release the capture
                                torch.cuda.empty_cache()  # Clear CUDA memory
                                st.stop()  # Stop streamlit app

                        # Display FPS in sidebar
                            fps_display.metric("FPS", f"{fps:.2f}") 
                        # Release the capture
                        cap.release()

                        
                        # Clear CUDA memory
                        torch.cuda.empty_cache()

                        # Destroy window
                        cv2.destroyAllWindows()

            # Web-cam
            elif options == 'Webcam':
                cam_options = st.sidebar.selectbox('Webcam Channel', ('Select Channel', '0', '1', '2', '3'))
                vid_file_name = int(cam_options)
  
                if not cam_options == 'Select Channel':
                    pred2 = st.sidebar.button("Start")
                    if pred2:
                        webrtc_streamer(
                    key="example",
                    video_transformer_factory=lambda: MyVideoTransformer(confidence, model),
                    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                    media_stream_constraints={"video": True, "audio": False},
                    )                                                                 
                
                
            # RTSP
           # elif options == 'RTSP':
            #    rtsp_url = st.sidebar.text_input(
            #        'RTSP URL:',
            #        'eg: rtsp://admin:name6666@198.162.1.58/cam/realmonitor?channel=0&subtype=0'
            #    )
            #    pred = st.sidebar.button("Start")
            #    cap = cv2.VideoCapture(rtsp_url)

# Main function call
if __name__ == "__main__":
    predict()
#
# if (cap != None) and pred:
#     stframe1 = st.empty()
#     stframe2 = st.empty()
#     stframe3 = st.empty()
#     while True:
#         success, img = cap.read()
#         if not success:
#             st.error(
#                 f"{options} NOT working\nCheck {options} properly!!",
#                 icon="🚨"
#             )
#             break

#         img, current_no_class = get_yolo(img, model, confidence, color_pick_list, class_labels, draw_thick)
#         st.image(img, channels='BGR')

#         # FPS
#         c_time = time.time()
#         fps = 1 / (c_time - p_time)
#         p_time = c_time
        
#         # Current number of classes
#         class_fq = dict(Counter(i for sub in current_no_class for i in set(sub)))
#         class_fq = json.dumps(class_fq, indent = 1)
#         class_fq = json.loads(class_fq)
#         df_fq = pd.DataFrame(class_fq.items(), columns=['Class', 'Quantity'])
        
#         # Updating Inference results
#         get_system_stat(stframe1, stframe2, stframe3, fps, df_fq)