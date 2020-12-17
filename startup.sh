#!/bin/bash
echo -e "\xf0\x9f\x8e\x89 \e[96mWelcome to the Wesley Docker container! \xf0\x9f\x8e\x89"
read -p "Do you want to start up Jupyter Notebook? [y/n]" answer
if [ "$answer" == y ] ; then
        echo -e "\e[92mStarting up Jupyter Notebook!\e[39m"
        jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root
else echo -e "\e[91mPlease use another IDE to work in the container (such as VS Code) \e[39m"
fi
