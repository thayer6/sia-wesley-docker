# this is a a script to start the Wesley container

# local directory
#mount_var="/Users/caseythayer/SIA/wesleytest"

# docker mount location
docker_loc="/docker_loc/"

# run container wesTest
docker run -it -d -v "${pwd}:${docker_loc}" -p 8888:8888 --name wesley siawesley:1.0

docker exec -t wesley bash

# run startup script
#./startup.sh

# to stop container run
# docker stop <container name>