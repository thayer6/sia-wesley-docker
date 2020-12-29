#!/bin/bash
# color definitions
COL_NORM="$(tput setaf 9)"
COL_RED="$(tput setaf 1)"
COL_GREEN="$(tput setaf 2)"
COL_CYAN="$(tput setaf 6)"
# welcome lines
echo -e "\xf0\x9f\x8e\x89 ${COL_CYAN} Welcome to the SIA Wesley Docker container!${COL_NORM} \xf0\x9f\x8e\x89"
echo "${COL_GREEN}This script starts up either Jupyter Notebook or Jupyter Lab in the Wesley Docker container${COL_NORM}"
# ask user if they want to run jupyter notebook
read -p "Do you want to start up Jupyter Notebook? [y/n]" answer1
if [ "$answer1" == y ] ; then # if yes then startup jupyter notebook
        echo -e "${COL_GREEN}Starting up Jupyter Notebook!${COL_NORM}" 
        jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root
elif [ "$answer1" == n ] ; then # if no ask if they want to start jupyter lab
        read -p "Do you want to start up Jupyer Lab? [y/n]" answer2
            if [ "$answer2" == y ] ; then # if yes start up jupyter lab
                echo "${COL_GREEN}Starting up Jupyter Lab!${COL_NORM}"
                jupyter lab --ip=0.0.0.0 --port=8888 --allow-root
            else echo -e "${COL_CYAN}Please use another IDE to work in the container or use the container bash directly.${COL_NORM}" # if user does not want jupyter notebook or jupyter lab use a different IDE
            fi
else echo "${COL_RED}Please enter a valid response. The input must be lower case and either 'y' or 'n'.${COL_NORM}" # ask user to provide case sensitive response for now -- to be updated with smarter input recognition
fi