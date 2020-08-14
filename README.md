# Um OCR simples com Keras (MNIST)

### Instale as bibliotecas necessárias

`pip install -r requirements`

### Treine o modelo com o MNIST dataset

`bin/train.py`

O modelo será salvo na pasta `models`.

### A imagem-teste base64 está no arquivo `base64image`.

### Para inicializar a API

`uvicorn main:app --reload`

Um arquivo de banco de dados `records.db` será criado.

### Acessando o FastAPI docs

`http://127.0.0.1:8000/docs`

### Executando GET /predict, uma previsão será realizada e salva como nova entrada no banco de dados.
