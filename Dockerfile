FROM python:3.7.3
WORKDIR /caseythayer/SIA/wesleytest/save_loc/

COPY requirements.txt .

RUN echo "installing requirements..."
RUN pip install -r requirements.txt
RUN pip install jupyter
RUN echo "done!"

COPY . .

RUN ["chmod", "+x", "./startup.sh"]
RUN ["chmod", "+x", "./wesley_build.sh"]
RUN ["chmod", "+x", "./wesley_run.sh"]

#ENTRYPOINT [ "./startup.sh"]