#!/bin/bash
# welcome lines
echo -e "\xf0\x9f\x8e\x89 Welcome to the Wesley Docker container! \xf0\x9f\x8e\x89"
echo "This script starts up either Jupyter Notebook or Jupyter Lab in the Wesley Docker container"
# ask user if they want to run jupyter notebook
read -p "Do you want to start up Jupyter Notebook? [y/n]" answer1
if [ "$answer1" == y ] ; then # if yes then startup jupyter notebook
        echo -e "Starting up Jupyter Notebook!" 
        jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root
elif [ "$answer1" == n ] ; then # if no ask if they want to start jupyter lab
        read -p "Do you want to start up Jupyer Lab? [y/n]" answer2
            if [ "$answer2" == y ] ; then # if yes start up jupyter lab
                echo "Starting up Jupyter Lab!"
                jupyter lab --ip=0.0.0.0 --port=8888 --allow-root
            else echo -e "Please use another IDE to work in the container or use the container bash directly." # if user does not want jupyter notebook or jupyter lab use a different IDE
            fi
else echo "Please enter a valid response. The input must be lower case and either 'y' or 'n'." # ask user to provide case sensitive response for now -- to be updated with smarter input recognition
fi