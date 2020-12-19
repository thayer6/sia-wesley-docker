# this is a a script to start the Wesley container
# make sure that you're in the directory you want to use the container with

# docker mount location -- nothing really goes in here but it's a connection to the local directory and the docker directory
docker_loc="/docker_loc/"

# run container wesley
docker run -it -d -v "${pwd}:${docker_loc}" -p 8888:8888 --name wesley siawesley:1.0

docker exec -t wesley bash

# run startup script
#./startup_jupyter.sh

# to stop container run
# docker stop wesley