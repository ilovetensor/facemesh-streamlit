import cv2
import streamlit as st
import mediapipe as mp
import numpy as np
import time
import tempfile
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode
import av

global prevTime
prevTime = 0



mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_drawing_styles = mp.solutions.drawing_styles 

DEMO_IMAGE = 'demo.jpg'
DEMO_VIDEO = 'demo.mp4'

st.title('Face Mesh App using Media Pipe')

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true] > div:first-child{
        width: 550px
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
        width: 350px
        margin-left: -350px
    }
    </style>
    """,
    unsafe_allow_html = True,
)

st.sidebar.title('FaceMesh Sidebar')
st.sidebar.subheader('parameters')

@st.cache_data()
def image_resize(image, width=None, height=None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image 
    
    if width is None:
        r = height/float(h)
        dim = (int(w*r), height)
    else:
        r = width/float(w)
        dim = (width, int(h*r))
 
    # resizee the image
    resized = cv2.resize(image, dim, interpolation=inter)

    return resized

app_mode = st.sidebar.selectbox('Choose the App Mode',
                                ['About App', 'Run on Image', 
                                 'Run on Webcam'],
                                )


if app_mode == 'About App':
    st.markdown('In this app lication we are using **MediaPipe** for creating a FaceMesh App, **Streamlit** is used to create the Web App GUI')
    

    st.markdown(
        '''
        Hey this is **Rohan Sharma**. Our team has built a streamlit webapp which applys a facemesh to any detected face using mediapipe lib. This facemesh can be applied to both image and live web cam video.
    
        - [GitHub](https://github.com/ilovetensor/facemesh-streamlit)\n
        
        '''
    )

elif app_mode == 'Run on Image':
    st.sidebar.markdown('---')
    
    st.markdown('**Detected Faces**')
    kpi1_text = st.markdown('0')

    max_faces = st.sidebar.number_input('Maximum Number of Faces', value=3, min_value=1)
    detection_confidence = st.sidebar.slider('Min Detection Conficence', min_value=0.0, max_value=1.0, value=0.5)
    st.markdown('---')
    try:
        img_file_buffer = st.sidebar.file_uploader("Upload an Image", type=['jpg', 'png', 'jpeg'])
    except:
        print("=====@@@@@=========")
    if img_file_buffer is not None:
        image = np.array(Image.open(img_file_buffer))

    else:
        demo_image = DEMO_IMAGE
        image = np.array(Image.open(demo_image))
    
    st.sidebar.text('Original Image')
    st.sidebar.image(image)

    face_count = 0
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=0)


    ## Dashboard

    with mp_face_mesh.FaceMesh(
    static_image_mode = True,
    max_num_faces = max_faces ,
    min_detection_confidence = detection_confidence) as face_mesh:
        
        results = face_mesh.process(image)
        out_image = image.copy()

        # FaceLandmark Drawing
        for face_landmarks in results.multi_face_landmarks:
            face_count += 1

            mp_drawing.draw_landmarks(
            image = out_image,
            landmark_list = face_landmarks,
            connections = mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec = drawing_spec)

            kpi1_text.write(f"<h1 style='text-align:center; color:red;padding: 0px;margin: 0px'>{face_count}</h1>", unsafe_allow_html=True)
        st.subheader('Output Image')
        st.image(out_image, use_column_width=True)


    

elif app_mode == 'abs':

    st.set_option('deprecation.showfileUploaderEncoding', False)



    


    max_faces = st.sidebar.number_input('Maximum Number of Faces', value=5, min_value=1)
    detection_confidence = st.sidebar.slider('Min Detection Conficence', min_value=0.0, max_value=1.0, value=0.5)
    tracking_confidence = st.sidebar.slider('Min Tracking Conficence', min_value=0.0, max_value=1.0, value=0.5)
    st.markdown('---')

    st.markdown('## Output')
    
    stframe = st.empty()
    video_file_buffer = st.sidebar.file_uploader('Upload a video', type=['mp4', 'mov', 'avi', 'mkv', 'asf', 'm4v0'])
    tffile = tempfile.NamedTemporaryFile(delete=False)

    if not video_file_buffer:
        vid = cv2.VideoCapture(DEMO_VIDEO)
        tffile.name = DEMO_VIDEO
        
    else:
        tffile.write(video_file_buffer.read())
        vid = cv2.VideoCapture(tffile.name)

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(vid.get(cv2.CAP_PROP_FPS))


    #  Recording Part
    # codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    # codec = cv2.VideoWriter_fourcc(*'mp4v')
    # fourcc = cv2.VideoWriter_fourcc(*'h264')
    # fourcc = cv2.VideoWriter_fourcc(*'H264')
    # out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc(*'MJPG'), fps_input, (width,height))
    # out = cv2.VideoWriter('output.mp4',fourcc, fps_input, (width,height))
    


    st.sidebar.text('Input Video')
    st.sidebar.video(tffile.name)

    fps = 0
    i = 0

    drawing_spec = mp_drawing.DrawingSpec(thickness=0, circle_radius=0)
    
    kpi1, kpi2, kpi3 = st.columns(3)

    with kpi1:
        st.markdown('**Frame Rage**')
        kpi1_text = st.markdown('0')
    with kpi2:
        st.markdown('**Frame Rage**')
        kpi2_text = st.markdown('0')
    with kpi3:
        st.markdown('**Frame Rage**')
        kpi3_text = st.markdown('0')

    st.markdown('---')

    
        






    face_count = 0
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


    ## Dashboard

    with mp_face_mesh.FaceMesh(
    max_num_faces = max_faces ,
    min_tracking_confidence = tracking_confidence,
    min_detection_confidence = detection_confidence
    ) as face_mesh:
        prevTime = 0

        while vid.isOpened():
            i += 1
            ret, frame = vid.read()
           

            if not ret:
                continue

            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # if record:
            #     # print('writing')
            # cv2.waitKey(1)
            





            results = face_mesh.process(frame)
            frame.flags.writeable = True

            face_count = 0
            if results.multi_face_landmarks:
                # Face Landmark Drawing
                for face_landmarks in results.multi_face_landmarks:
                    face_count += 1

                    mp_drawing.draw_landmarks(
                    image = frame,
                    landmark_list = face_landmarks,
                    connections = mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec = drawing_spec,
                    connection_drawing_spec = drawing_spec)

            currTime = time.time()
            fps = 1/(currTime - prevTime)
            prevTime = currTime
            
            # out.write(frame)  


            kpi1_text.write(f"<h1 style='text-align:center; color:red;padding: 0px;margin: 0px'>{int(fps)}</h1>", unsafe_allow_html=True)
            kpi2_text.write(f"<h1 style='text-align:center; color:red;padding: 0px;margin: 0px'>{face_count}</h1>", unsafe_allow_html=True)
            kpi3_text.write(f"<h1 style='text-align:center; color:red;padding: 0px;margin: 0px'>{width}</h1>", unsafe_allow_html=True)
        
            frame = cv2.resize(frame, (0,0), fx=0.8, fy=0.8)
            frame = image_resize(image= frame, width = 640)
            stframe.image(frame, channels = 'BGR', use_column_width=True)
            
          


           
    
              
    vid.release()
   
    cv2.destroyAllWindows()
    

    
 
  
    
    # st.text('Video Processed')
    # video_file = open('output.mp4', 'rb')
    # video_bytes = video_file.read()

    # st.video(video_bytes)









if app_mode =='Run on Webcam':
    st.title('live video feed is here')
    
    



    
    prevTime = 0
    fps_ = 0
    faces = 0
    width = 0

    # Dashboard
    st.sidebar.markdown('---')
    max_faces = st.sidebar.number_input('Max Face Count', value=2, min_value=1)
    detection_confidence = st.sidebar.slider('Min. Detection Confidence', 0.0, 1.0, 0.10)

    st.markdown('---')

    def callback(frame):
        global prevTime
        
        img = frame.to_ndarray(format='bgr24')

        output, prevTime = apply_facemesh(img, prevTime)

        return av.VideoFrame.from_ndarray(output, format='bgr24')
        
            
    def apply_facemesh( image, prevTime):
        
        # image = image

        face_count = 0
        drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


        ## Image Processing

        with mp_face_mesh.FaceMesh(
        static_image_mode = True,
        max_num_faces = max_faces ,
        min_detection_confidence = detection_confidence
        ) as face_mesh:
            # print(max_faces, detection_confidence)
            
            results = face_mesh.process(image)
            out_image = image.copy()
            
            # FaceLandmark Drawing
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    face_count += 1

                    mp_drawing.draw_landmarks(
                    image = out_image,
                    landmark_list = face_landmarks,
                    connections = mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec = drawing_spec)

            currTime = time.time()
            fps = 1/ (currTime - prevTime)
            prevTime = currTime
        
            return out_image, prevTime

    RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

    webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION,
                        video_frame_callback=callback)

