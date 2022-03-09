import take_and_recog
import tensorflow as tf

def TwoDConverToOneD(List):
    new_list = []
    for line in List:
        new_list.append(line[1])
    return new_list

if __name__ == '__main__':
    model = tf.keras.models.load_model("models/number_rcog_mobilenet.h5")
    print(TwoDConverToOneD(take_and_recog.photo_take_and_recog(model)))
