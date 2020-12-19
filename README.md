# Wesley Docker Container
This is the code used to build and run the Wesley Docker Container!

How to get started:
1. Navigate to the directory where you'd like to work locally on Wesley
    In the terminal this can be done using the cd command to traverse your local directories (i.e. cd /foldername/). You can also create a new directory using the mkdir command (i.e. mkdir welseyfolder).
2. Clone the Wesley repository of interest and enter the directory

    git clone Wesley-repo-URL
    
    cd repo-directory
3. Build image using the following command
    docker build -t siawesley:1.0 .
4. Run the container with ./wesley.sh
5. You are now in the container bash. Type exit to leave the bash and then run docker stop wesley

How to get up and running during development using Jupyter Notebook or Jupyter Lab:
1. Navigate to the directory where the cloned Wesley repository resides (i.e. cd /wesleycode/)
2. Start the Docker container with the following command: 
    docker start wesley
3. You are now in the container bash. Run ./startup_jupyter.sh and follow the prompts to start working in Jupyter Notebook or Jupyter Lab
4. When you're done run the following command to stop the container:
    docker stop wesley

How to get up and running during development using VS Code: TBD
