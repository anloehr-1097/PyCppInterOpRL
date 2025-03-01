VOLUMES="-v /Users/Andy/Code/ML-Practice/Cpp:/home/"
COMMANDS="/bin/bash"
# docker run -it $VOLUMES --name cpp_dev_container cpp_dev
 #docker container start cpp_dev_container

if [ "$#" -ne 1 ]; then
    echo "Usage: run_container.sh [command]"
    echo "Commands:"
    echo "  start: Start the container"
    echo "  build: Build the container"
    echo "  login: Log into the container"
    echo "  stop: Stop the container"
    echo "  delete: Delete the container"
fi


if [ "$1" == "build" ]; then
    docker compose up -d --build
    docker logs -f cpp_dev_container
elif [ "$1" == "login" ]; then
    docker exec -it cpp_dev_container /bin/bash
    # docker run -it $VOLUMES --name cpp_dev_container cpp_dev $COMMANDS
elif [ "$1" == "start" ]; then
    docker container start cpp_dev_container
elif [ "$1" == "stop" ]; then
    docker container stop cpp_dev_container
    # docker run -it $VOLUMES --name cpp_dev_container cpp_dev $COMMANDS
elif [ "$1" == "delete" ]; then
    docker compose down
    # docker run -it $VOLUMES --name cpp_dev_container cpp_dev $COMMANDS
else 
    echo "Invalid command"
fi
