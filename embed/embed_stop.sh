#!/bin/bash

# Prefix to filter container names
PREFIX="embed-"

# Get all container names starting with the specified prefix and store them in an array
mapfile -t CONTAINER_NAMES < <(docker ps -a --format '{{.Names}}' | grep "^${PREFIX}")

echo "Stopping containers: ${CONTAINER_NAMES[@]}"

# Iterate over the array and print each container name
for name in "${CONTAINER_NAMES[@]}"; do
    docker stop "${name}" 2>/dev/null || true
    echo "Stopped container ${name}"
done

# Wait for all containers to be completely stopped
for name in "${CONTAINER_NAMES[@]}"; do
    while [ "$(docker ps -a -f name=${name} --format '{{.Status}}' | grep -v 'Exited')" ]; do
        sleep 1
    done
done

echo "Stopped all containers"