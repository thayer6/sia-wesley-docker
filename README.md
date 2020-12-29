# Wesley Docker Container
This is the code used to build and develop in the Wesley Docker Container! There are two workflows, one for using the Jupyter Notebook or Jupyter Lab and another for using VS Code. Once you start with one workflow, you are not stuck with it, you can use both or either workflow when developing Wesley. If you have any questions or run into any errors, please don't hesitate to reach out to Casey and/or Terrence!

# Jupyter Workflow

How to get started:
1. Navigate to the directory where you'd like to work locally on Wesley
    In the terminal this can be done using the cd command to traverse your local directories (i.e. cd /foldername). You can also create a new directory using the mkdir command (i.e. mkdir welseyfolder).
2. Clone the Wesley repository of interest and enter the directory

    git clone <"wesley-repo-url">
    
    cd wesleyfolder

3. Create your own config shell script off of the wesley_example.sh that exists in the repo.

    Create a new shell script

    touch wesley.sh

    Copy the contents from the example script into your new shell script
    cp wesley_example.sh wesley.sh

    Then edit the mount_loc variable to contain the full path to your working directory (ie. where you cloned the SIA Wesley repository)


3. Pull and build the image using the following command

    docker pull thayer6/wesleydocker:1.0

4. Create and start the container with by running ./wesley.sh
5. You are now in the container bash. To start coding right away in jupyter notebook or jupyter lab skip to step 3 below. 
    
How to get up and running:
1. In the terminal, navigate to the directory where you'd like to work on Wesley (i.e. cd /wesleyfolder/)
2. Start the Docker container with the following command (if not already started): 

    docker start wesley

3. To enter the wesley container bash use the following command:

    docker exec -ti wesley bash

3. You are now in the container bash. Run ./startup_jupyter.sh and follow the prompts to start working in Jupyter Notebook or Jupyter Lab

4. When you're done run the following command to stop the container:

    docker stop wesley

# VS Code Workflow
How to get started:
1. Navigate to the directory where you'd like to work locally on Wesley
    In the terminal this can be done using the cd command to traverse your local directories (i.e. cd /foldername). You can also create a new directory using the mkdir command (i.e. mkdir welseyfolder).
2. Clone the Wesley repository of interest and enter the directory

    git clone <"wesley-repo-url">
    
    cd wesleyfolder

How to get up and running:
1. Open VS Code
    Download the "Remote Containers" extension
2. Click on the arrows in the bottom left hand corner and select "Open folder in remote container"
    Navigate to the folder you want to work on Wesley in and click open
3. The container will now build off of the .devcontainer folder and you can develop/run all your code within the container. Additionally, you can push/pull directly to a github repo from the remote container in VS Code!