import numpy as np
import cv2
import streamlit as st
#Creating title for Streamlit app
st.title("Fabric Defect Detection with OpenCV")
#uploading file for processing
app_mode = st.sidebar.selectbox('Choose the App Mode',
                                ['About App','Run on Image','Run on Video','Run on WebCam'])

if app_mode == 'About App':
    st.subheader("About")
    st.markdown("<h5>This is the Flame Detection App created with custom trained models using YoloV5</h5>",unsafe_allow_html=True)
    
    st.markdown("- <h5>Select the App Mode in the SideBar</h5>",unsafe_allow_html=True)
    st.image("Images/first_1.png")
    st.markdown("- <h5>Upload the Image and Detect the Fires in Images (For example, 0.6 means 60% of Unburned Carbon)</h5>",unsafe_allow_html=True)
    st.image("Images/second_2.png")
    st.markdown("- <h5>Upload the Video and Detect the fires in Videos (For example, 0.6 means 60%)</h5>",unsafe_allow_html=True)
    st.image("Images/third_3.png")
    st.markdown("- <h5>Live Detection</h5>",unsafe_allow_html=True)
    st.image("Images/fourth_4.png")
    st.markdown("- <h5>Click Start to start the camera</h5>",unsafe_allow_html=True)
    st.markdown("- <h5>Click Stop to stop the camera</h5>",unsafe_allow_html=True)
    
    st.markdown("""
                ## Features
- Detect on Image
- Detect on Videos
- Live Detection
## Tech Stack
- Python
- PyTorch
- Python CV
- Streamlit
- YoloV5

  🔗 Links
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](http://burnhancer.com/)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://ir.linkedin.com/company/burnhancer-discover-incorporation)
[![twitter](https://img.shields.io/badge/Github-1DA1F2?style=for-the-badge&logo=github&logoColor=white)](https://github.com/peymanbayat)
""")   

if app_mode == 'Run on Image':

 uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
 if uploaded_file is not None:
    # Read the uploaded image using OpenCV
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)

    img=image.copy()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    blur=cv2.blur(gray,(10,10))

    dst=cv2.fastNlMeansDenoising(blur,None,10,7,21)

    _,binary=cv2.threshold(dst,127,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    kernel=np.ones((5,5),np.uint8)

    erosion=cv2.erode(binary,kernel,iterations=1)
    dilation=cv2.dilate(binary,kernel,iterations=1)

    if (dilation==0).sum()>1:
        contours,_=cv2.findContours(dilation,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        for i in contours:
            if cv2.contourArea(i)<261121.0:
                cv2.drawContours(img,i,-1,(0,0,255),3)
            cv2.putText(img,"defective fabric",(30,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    else:
        cv2.putText(img, "Good fabric", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    st.image(image,caption="original image",channels="BGR")
    st.image(blur,caption="blur")
    st.image(binary,caption="binary")
    st.image(erosion,caption="erosion")
    st.image(dilation,caption="dilation")
    st.image(img,caption="defected area",channels="BGR")
cv2.waitKey(0)
cv2.destroyAllWindows()


if app_mode == 'Run on Video':
    st.subheader("No. of Required Oxygen Valve(s):")
    text = st.markdown("")
    
    st.sidebar.markdown("---")
    
    st.subheader("Output Image ")
    stframe = st.empty()
    
    #Input for Video
    video_file = st.sidebar.file_uploader("Upload a Video",type=['mp4','mov','avi','asf','m4v'])
    st.sidebar.markdown("---")
    tffile = tempfile.NamedTemporaryFile(delete=False)
    
    if not video_file:
        vid = cv2.VideoCapture(demo_video)
        tffile.name = demo_video
    else:
        tffile.write(video_file.read())
        vid = cv2.VideoCapture(tffile.name)
    
    st.sidebar.markdown("**Input Video**")
    st.sidebar.video(tffile.name)
    
    # predict the video
   # while vid.isOpened():
    #    ret, frame = vid.read()
     #   if not ret:
      #      break
       # frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
       # model = load_model()
       # results = model(frame)
       # length = len(results.xyxy[0])
       # output = np.squeeze(results.render())
       # text.write(f"<h1 style='text-align: center; color:blue;'>{length}</h1>",unsafe_allow_html = True)
       # stframe.image(output)
      uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
 if uploaded_file is not None:
    # Read the uploaded image using OpenCV
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)

    img=image.copy()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    blur=cv2.blur(gray,(10,10))

    dst=cv2.fastNlMeansDenoising(blur,None,10,7,21)

    _,binary=cv2.threshold(dst,127,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    kernel=np.ones((5,5),np.uint8)

    erosion=cv2.erode(binary,kernel,iterations=1)
    dilation=cv2.dilate(binary,kernel,iterations=1)

    if (dilation==0).sum()>1:
        contours,_=cv2.findContours(dilation,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        for i in contours:
            if cv2.contourArea(i)<261121.0:
                cv2.drawContours(img,i,-1,(0,0,255),3)
            cv2.putText(img,"defective fabric",(30,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    else:
        cv2.putText(img, "Good fabric", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    st.image(image,caption="original image",channels="BGR")
    st.image(blur,caption="blur")
    st.image(binary,caption="binary")
    st.image(erosion,caption="erosion")
    st.image(dilation,caption="dilation")
    st.image(img,caption="defected area",channels="BGR")
cv2.waitKey(0)
cv2.destroyAllWindows()
   
if app_mode == 'Run on WebCam':
    st.subheader("No. of Required Oxygen Valve(s):")
    text = st.markdown("")
    
    st.sidebar.markdown("---")
    
    st.subheader("Output")
    stframe = st.empty()
    
    run = st.sidebar.button("Start")
    stop = st.sidebar.button("Stop")
    st.sidebar.markdown("---")
    
    cam = cv2.VideoCapture(0)
    if(run):
        while(True):
            if(stop):
                break
            ret,frame = cam.read()
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            model = load_model()
            results = model(frame)
            length = len(results.xyxy[0])
            output = np.squeeze(results.render())
            text.write(f"<h1 style='text-align: center; color:blue;'>{length}</h1>",unsafe_allow_html = True)
            stframe.image(output)
