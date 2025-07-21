### TEI only
curl 127.0.0.1:8080/multilingual-e5-large-instruct     -X POST     -d '{"inputs":"What is Deep Learning?"}'     -H 'Content-Type: application/json'
curl 127.0.0.1:8080/UAE-Large-V1     -X POST     -d '{"inputs":"What is Deep Learning?"}'     -H 'Content-Type: application/json'


### API
# TEI
curl 127.0.0.1:5000/compute_embedding      -X POST     -d '{"model_name": "multilingual-e5-large-instruct", "sentences":["What is Deep Learning?"]}'     -H 'Content-Type: application/json'
curl 127.0.0.1:5000/compute_embedding      -X POST     -d '{"model_name": "UAE-Large-V1", "sentences":["What is Deep Learning?"]}'     -H 'Content-Type: application/json'

# sentence_transformer
curl 0.0.0.0:5000/compute_embedding      -X POST     -d '{"model_name": "SBERT-bert-base-spanish-wwm-uncased", "sentences":["What is Deep Learning?"]}'     -H 'Content-Type: application/json'
curl 0.0.0.0:5000/compute_embedding      -X POST     -d '{"model_name": "LaBSE", "sentences":["What is Deep Learning?"]}'     -H 'Content-Type: application/json'
# curl 0.0.0.0:5000/compute_embedding     -X POST     -d '{"inputs":"What is Deep Learning?"}'     -H 'Content-Type: application/json'