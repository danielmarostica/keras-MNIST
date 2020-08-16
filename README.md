# Uma rede neural Keras em API

### Instale as bibliotecas necessárias

`pip install -r requirements`

### Treine o modelo com o MNIST dataset

Execute o arquivo `bin/train.py`.

O modelo será salvo na pasta `models`. O repositório conta com um modelo pré-treinado (val_acc ~ 97%), mas pode atingir 99% com 50 epochs.

### Para inicializar a API

Na pasta `bin`, execute `uvicorn main:app --reload`

Um arquivo de banco de dados `records.db` será criado.

A imagem-teste está no arquivo `base64image`. É o número 4 codificado em base64.

### Acessando o FastAPI docs

Acesse `http://127.0.0.1:8000/docs`

### Executando GET /predict, uma previsão será realizada e salva como nova entrada no banco de dados.
