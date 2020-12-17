FROM python:3.7.3
WORKDIR /caseythayer/SIA/envtest

COPY requirements.txt .

RUN pip install -r requirements.txt
RUN pip install jupyter

COPY . .

RUN ["chmod", "+x", "./letsgo.sh"]

ENTRYPOINT [ "./letsgo.sh"]