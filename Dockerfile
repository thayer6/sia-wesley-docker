FROM python:3.7.3
WORKDIR /caseythayer/SIA/envtest

COPY requirements.txt .

RUN pip install -r requirements.txt
RUN pip install jupyter

COPY . .

RUN ["chmod", "+x", "./startup.sh"]

ENTRYPOINT [ "./startup.sh"]