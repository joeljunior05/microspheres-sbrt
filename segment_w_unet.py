from functools import partial

import numpy as np

from utils.inout import InOutLoop
from utils.imgproc import read_img, write_img, copy_border, remove_border

STEP = 16
SIZE = 128

def crop_img(img):
    rows, cols, _ = img.shape
    ret = []

    for c in range(0, cols - SIZE, STEP):
        for r in range(0, rows - SIZE, STEP):
            ret.append(img[r:r+SIZE, c:c+SIZE, :3])
    return ret


def recover_cropped_img(img, predicted):
    rows, cols, _ = img.shape
    ret_img = np.zeros((rows, cols))
    sum_img = np.ones((rows, cols))

    idx = 0
    for c in range(0, cols - SIZE, STEP):
        for r in range(0, rows - SIZE, STEP):
            ret_img[r:r+SIZE, c:c +
                    SIZE] = np.add(ret_img[r:r+SIZE, c:c+SIZE], predicted[idx])
            sum_img[r:r+SIZE, c:c +
                    SIZE] = np.add(sum_img[r:r+SIZE, c:c+SIZE], np.ones((SIZE, SIZE)))
            idx += 1

    return ret_img / sum_img


def generate_validation_dataset(image):
    validation = np.array(crop_img(image))
    validation = validation.astype(np.uint8)
    return validation

def read_img_border(filename_img, padding=0):
        out_img = read_img(filename_img)

        if padding > 0:
            out_img[0] = copy_border(out_img[0], padding)

        return out_img

def write_img_border(filename_img, image, padding=0):
        out_img = image
    
        if padding > 0:
            out_img = remove_border(out_img, padding)

        write_img(filename_img, out_img)

def load_keras_unet_model(modelfile, weightsfile):
    from keras.models import model_from_json

    # load json and create model
    json_file = open(modelfile, 'r')
    loaded_json_file = json_file.read()
    loaded_model = model_from_json(loaded_json_file)
    json_file.close()

    # load weights into new model
    loaded_model.load_weights(weightsfile)
    loaded_model.compile(loss='binary_crossentropy',
                         optimizer='rmsprop', metrics=['accuracy'])
    print("Loaded model from disk")

    return loaded_model

def predict(input, model=None, debug_fn=None):
    input_image = input[0]

    validation = generate_validation_dataset(input_image)

    print("Predicting output")
    predicted_output = model.predict(validation)

    if debug_fn:
        debug_fn(predicted_output[0])

    return recover_cropped_img(input_image, predicted_output)

if __name__ == "__main__":

    inputs_folder = "dataset/prediction/inputs"
    outputs_folder = "dataset/prediction/outputs_unet"

    model_filename = "trained_model/model.json"
    model_weights_filename = "trained_model/weights.h5"

    mainloop = InOutLoop(input_folder=inputs_folder, output_folder=outputs_folder,
                         extensions=['jpeg', 'png', 'jpg'])

    model = load_keras_unet_model(model_filename, model_weights_filename)

    mainloop.on_input(partial(read_img_border, padding=64))
    mainloop.on_run(partial(predict, model=model, debug_fn=mainloop.log))
    mainloop.on_output(partial(write_img_border, padding=64))

    mainloop.run()