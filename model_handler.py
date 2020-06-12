from keras.models import model_from_json


class ModelHandler:
    """Utility class; handles machine learning models."""
    def save_model(self, model, stock):
        model_json = model.to_json()
        with open(f'{stock}_model.json', 'w') as json_file:
            json_file.write(model_json)
        model.save_weights(f'{stock}_model.h5')
        print("Saved model to disk")


    def load_model(self, stock):
        json_file = open(f'{stock.ticker}_model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights(f'{stock.ticker}_model.h5')
        print("Loaded model from disk")

        return model
