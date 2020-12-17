# Wesley
This is the core code that the SIA Data Science Team uses for Wesley.

To build run:
docker build -t siawesley:1.0 .
Where siawesley is the name of the image and 1.0 is the tag

After building:
- change the path in volume section of docker-compose.yml to a path accessible on your machine to connect to Jupyter Notebook
- run the container using:
docker run -it siawesley:1.0 .
The -it tag allows the executable to run where you can choose to run with Jupyter Notebook or not. Here is where I can't seem to get connected to the Jupyter Notebook site.


There may be another way around this using docker-compose and adding the startup executable as a command in that file but I haven't been able to run it interactively yet