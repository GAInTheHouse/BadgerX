FROM tensorflow/tensorflow:nightly
WORKDIR /root

RUN pip install pandas google-cloud-storage scikit-learn numpy fsspec gcsfs tables

RUN apt-get update; apt-get install curl -y

RUN pip install cloudml-hypertune

RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg  add - && apt-get update -y && apt-get install google-cloud-sdk -y

ENV PATH $PATH:/root/tools/google-cloud-sdk/bin

RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg

COPY util.py ./util.py
COPY task.py ./task.py

ENTRYPOINT ["python", "task.py"]
