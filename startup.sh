#!/bin/bash
echo -e "\xf0\x9f\x8e\x89 Welcome to the Wesley Docker container! \xf0\x9f\x8e\x89"
read -p "Do you want to start up Jupyter Notebook? [y/n]" answer
if [ "$answer" == y ] ; then
        echo -e "Starting up Jupyter Notebook!"
        jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root
else echo -e "Please use another IDE to work in the container (such as VS Code)"
# jupyter lab --ip=0.0.0.0 --port=8888 --allow-root
fi
