FROM python:3.7.3
WORKDIR /SIA/wesley/

COPY requirements.txt .

RUN echo "installing requirements..."
RUN pip install -r requirements.txt
RUN pip install jupyter
RUN echo "done!"

COPY . .

RUN ["chmod", "+x", "./startup_jupyter.sh"]
RUN ["chmod", "+x", "./wesley.sh"]