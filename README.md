# Uma rede neural convolucional Keras em API

## Online

A API recebe uma string (imagem base64) como post request. Copie do arquivo `base64image` um exemplo (número 4).

[Acesse aqui.](https://keras-mnist-roit.herokuapp.com/docs)

A API pode demorar alguns segundos para iniciar no cloud, apos o post request.

## Local

### Instale as bibliotecas necessárias

`pip install -r requirements-local`

### Treine o modelo com o MNIST dataset

Execute o arquivo `models/train.py` para treinar um novo modelo, que será salvo na mesma pasta.

O repositório conta com um modelo pré-treinado (val_acc ~ 99%). Early stop em 16 epochs.

### Para inicializar a API

Execute `uvicorn main:app --reload`

### Acessando o FastAPI docs

Acesse `http://127.0.0.1:8000/docs`

Executando `GET /predict`, uma previsão será realizada e salva como nova entrada no banco de dados.

Um arquivo de banco de dados `records.db` será criado e incrementado a cada post request.
