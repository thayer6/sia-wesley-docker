FROM python:3.7.3
# this is the working directory in the docker container where local files will be copied to
WORKDIR /SIA/wesley/

# copy the requirements txt file
COPY requirements.txt .

# install requirements
RUN echo "installing requirements..."
RUN pip install -r requirements.txt
RUN pip install jupyter
RUN echo "done!"

# copy files from local directory into docker working directory
COPY . .

# give permissions to shell script files
RUN ["chmod", "+x", "./startup_jupyter.sh"]
RUN ["chmod", "+x", "./wesley.sh"]