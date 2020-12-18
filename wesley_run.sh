# this is a a script to start the Wesley container

# local directory
mount_var="/Users/caseythayer/SIA/wesleytest"

# docker mount location
save_loc="/save_loc/"

# run container wesTest
docker run -it -d -v "${mount_var}:${save_loc}" -p 8888:8888 --name wesTest siawesley:1.0

docker exec -t wesTest bash

# run startup script
#./startup.sh

# to stop container run
# docker stop <container name>