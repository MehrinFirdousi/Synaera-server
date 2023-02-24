FROM python:3.10-slim
COPY ./server.py /deploy/
COPY ./requirements.txt /deploy
COPY ./test8.h5 /deploy/
WORKDIR /deploy/
RUN pip install -r requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y
EXPOSE 80
ENTRYPOINT ["python", "server.py"]