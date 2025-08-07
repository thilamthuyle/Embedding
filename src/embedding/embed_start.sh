#!/bin/bash

# Default list of model names
# default_models=("all-MiniLM-L6-v2" "mxbai-embed-large-v1" "sentence-camembert-large")
default_models=("multilingual-e5-large-instruct" "UAE-Large-V1")

# Check if the models are given in the argument: ./process.sh model_1 model_2 model_3
if [ $# -eq 0 ]; then
    models=("${default_models[@]}")
else
    models=("$@")
fi

# Print list of models
echo "Models to process: ${models[@]}"

PREFIX="embed-"
NET_NAME="embedding-net"

# launch embed_stop.sh in the same directory as the current script
DIR="$(dirname "$(realpath "$0")")"
echo "DIR = $DIR"
sudo chmod +x "$DIR/embed_stop.sh"
sudo chmod +x "$DIR/embed_stoppost.sh"
# Stop the containers by passing the container names as arguments
$DIR/embed_stop.sh
$DIR/embed_stoppost.sh


# Check if there are GPUs on the machine
num_gpus=$(lspci | grep -ci vga)

# Use the appropriate Docker image based on the number of GPUs
if [ $num_gpus -eq 0 ]; then
    TEI_DOCKER_IMAGE="ghcr.io/huggingface/text-embeddings-inference:cpu-1.5"
    TEI_EXTRA_ARGS=""
else
    TEI_DOCKER_IMAGE="ghcr.io/huggingface/text-embeddings-inference:turing-1.5"
    TEI_EXTRA_ARGS="--gpus all"
fi

echo "TEI_DOCKER_IMAGE = $TEI_DOCKER_IMAGE"
echo "TEI_EXTRA_ARGS = $TEI_EXTRA_ARGS"


# Create the Docker network if it doesn't already exist
docker network ls | grep -q ${NET_NAME} || docker network create ${NET_NAME}


# Loop through the list of model names and process each one
for model_name in "${models[@]}"; do
    echo "Processing $model_name..."
    # Custom arguments for specific models
    # For sentence-camembert-*, we need to add the --pooling argument
    # Check if model_name contains "sentence-camembert"
    if [[ $model_name == "sentence-camembert-"* ]]; then
        MODEL_ARGS="--pooling mean"
    else
        MODEL_ARGS=""
    fi

    volume=/www/Embedding/model_zoo 
    docker run $TEI_EXTRA_ARGS --net ${NET_NAME} --name ${PREFIX}${model_name} -v $volume:/data --pull always --rm -d $TEI_DOCKER_IMAGE --model-id /data/${model_name} ${MODEL_ARGS} 
    echo "Started container for $model_name"
done



# Generate the dynamic embed_nginx.conf file
NGINX_CONF="$DIR/embed_nginx.conf"
touch $NGINX_CONF
# Make the file writable
sudo chmod 777 $NGINX_CONF

# Start writing the upstream configuration
cat <<EOL > $NGINX_CONF
EOL


# Loop through the model names again to append to the NGINX configuration
for model_name in "${models[@]}"; do
    cat <<EOL >> $NGINX_CONF
upstream ${model_name} {
    server ${PREFIX}${model_name};
}

EOL
done

# Append the server configuration to the NGINX configuration
cat <<EOL >> $NGINX_CONF
server {
EOL

for model_name in "${models[@]}"; do
    cat <<EOL >> $NGINX_CONF
    location /${model_name} {
        proxy_pass http://${PREFIX}${model_name}/;
    }
EOL
done

# Close the server block
cat <<EOL >> $NGINX_CONF
}
EOL

# Verify that all model containers are running before starting nginx
for model_name in "${models[@]}"; do
    until [ "`/usr/bin/docker inspect -f {{.State.Running}} ${PREFIX}${model_name}`" == "true" ]; do
        sleep 0.1;
    done;
done;

# Create a directory for custom NGINX HTML files
CUSTOM_HTML_DIR="$DIR/nginx_html"
mkdir -p "$CUSTOM_HTML_DIR"

# Create a default index.html file if it doesn't exist
if [ ! -f "$CUSTOM_HTML_DIR/index.html" ]; then
    echo "<html><body><h1>Welcome to the Embedding Service</h1></body></html>" > "$CUSTOM_HTML_DIR/index.html"
fi

# Create a default favicon.ico file if it doesn't exist
if [ ! -f "$CUSTOM_HTML_DIR/favicon.ico" ]; then
    echo -n "" > "$CUSTOM_HTML_DIR/favicon.ico"  # Empty favicon
fi

# Start the NGINX container
docker run -v ${NGINX_CONF}:/etc/nginx/conf.d/default.conf:ro \
           -v ${CUSTOM_HTML_DIR}:/usr/share/nginx/html:ro \
           -p 8080:80 --net ${NET_NAME} --name ${PREFIX}nginx nginx

# Verify the NGINX container is running
until [ "`/usr/bin/docker inspect -f {{.State.Running}} ${PREFIX}nginx`" == "true" ]; do
    echo "Waiting for NGINX container to start..."
    sleep 0.5
done

echo "Started container for nginx"


