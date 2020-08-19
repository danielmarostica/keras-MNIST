def load_model(name: str='model'):
    # loads trained model
    from tensorflow.keras.models import model_from_json
    with open('models/{}.json'.format(name), 'r') as json_file:
        cnn_json = json_file.read()
    cnn = model_from_json(cnn_json)
    cnn.load_weights('models/{}.h5'.format(name))
    
    return cnn

if __name__ == "__main__":
    load_model()
