FROM continuumio/anaconda3
COPY . /user/app 
EXPOSE 5000
WORKDIR /user/app
RUN pip install -r requirements.txt && \ 
    pip install streamlit && \
    apt update && apt install -y libsm6 libxext6 && \
    pip uninstall opencv-python && \
    pip uninstall opencv-contrib-python && \
    pip uninstall opencv-contrib-python-headless && \
    pip3 install opencv-python-headless
    
CMD streamlit run main.py