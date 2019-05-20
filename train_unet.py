import numpy as np
from keras.models import model_from_json

def image_generator(data, target, batch_size):
    """
    """ 
    L, W = data[0].shape[0], data[0].shape[1]
    while True:
        for i in range(0, len(data), batch_size):
            d, t = data[i:i + batch_size].copy(), target[i:i + batch_size].copy()

            for j in np.where(np.random.randint(0, 2, batch_size) == 1)[0]:
                d[j], t[j] = np.fliplr(d[j]), np.fliplr(t[j]) 
            for j in np.where(np.random.randint(0, 2, batch_size) == 1)[0]:
                d[j], t[j] = np.flipud(d[j]), np.flipud(t[j])

            npix = 15
            h = np.random.randint(-npix, npix + 1, batch_size)    
            v = np.random.randint(-npix, npix + 1, batch_size)    
            r = np.random.randint(0, 4, batch_size)               
            for j in range(batch_size):
                d[j] = np.pad(d[j], ((npix, npix), (npix, npix), (0, 0)),
                              mode='constant')[npix + h[j]:L + h[j] + npix,
                                               npix + v[j]:W + v[j] + npix, :]
                t[j] = np.pad(t[j], (npix,), mode='constant')[npix + h[j]:L + h[j] + npix, 
                                                              npix + v[j]:W + v[j] + npix]
                d[j], t[j] = np.rot90(d[j], r[j]), np.rot90(t[j], r[j])
            yield (d, t)


def load_keras_unet_model(model_filename, weights_filename):
    try:
        # load json and create model
        json_file = open(model_filename, 'r')
        loaded_json_file = json_file.read()
        loaded_model = model_from_json(loaded_json_file)
        json_file.close()
        
        # load weights into new model
        loaded_model.load_weights(weights_filename)
        loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        print("Loaded model from disk")

        return loaded_model
    except:
        return None

def train(samples, targets, model_filename, weights_filename, nb_epochs, bs):
    model = load_keras_unet_model(model_filename, weights_filename)
    print(model_filename, weights_filename)
    print(model)
    n_samples = len(samples)

    max_samples = int(n_samples / bs) * bs

    if max_samples == 0:
        max_samples = n_samples

    for nb in range(nb_epochs):
        print(nb, '/', nb_epochs)

        history = model.fit_generator(
            image_generator(samples[:max_samples], targets[:max_samples], batch_size=bs),
            steps_per_epoch=n_samples, epochs=1, verbose=1)

        print(history.history['acc'])

    return model

def save_model(model, model_filename, weights_filename):
    model_json = model.to_json()

    with open(model_filename, "w") as json_file:
        json_file.write(model_json)

    model.save_weights(weights_filename)

samples = np.load("dataset/trainning/samples.npy")
targets = np.load("dataset/trainning/targets.npy")

model_filename = "model/model.json"
model_weights_filename = "model/weights.h5"

model = train(samples, targets, model_filename, model_weights_filename, 32, 16)

out_model_filename = "trained_model/model.json"
out_model_weights_filename = "trained_model/weights.h5"

save_model(model, out_model_filename, out_model_weights_filename)