Run "docker build -t adsb_pynqz2 -f docker/Dockerfile ." in root dir to build docker

Run "docker run -p 8888:8888 -v /path/to/your/local/directory:/home/jovyan/work -it adsb_pynqz2" to run docker, replace "/path/to/your/local/directory" with your own path

docker run -p 8888:8888 -v /home/david/Bachelor/adsb_pynqz2:/home/jovyan/work -it adsb_pynqz2