# Uma rede neural convolucional Keras em API

## Online

A API recebe uma string (imagem base64) como post request. Copie do arquivo `base64image` um exemplo (número 4).

[Acesse aqui.](https://keras-mnist-roit.herokuapp.com/docs)

## Local

### Instale as bibliotecas necessárias

`pip install -r requirements`

### Treine o modelo com o MNIST dataset

Clone este repositório e execute o arquivo `modules/train.py` para treinar um novo modelo.

O modelo será salvo na pasta `models`. O repositório conta com um modelo pré-treinado (val_acc ~ 97%), mas pode atingir 99% com 50 epochs.

### Para inicializar a API

Execute `uvicorn main:app --reload`

### Acessando o FastAPI docs

Acesse `http://127.0.0.1:8000/docs`

Executando `GET /predict`, uma previsão será realizada e salva como nova entrada no banco de dados.

Um arquivo de banco de dados `records.db` será criado e incrementado a cada post request.
