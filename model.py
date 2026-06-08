import tensorflow as tf
from tensorflow.keras import layers, models
# print(tf.sysconfig.get_build_info())
def inception_block(x, filters):

    branch1 = layers.MaxPooling1D(pool_size=3, strides=1, padding="same")(x)
    branch1 = layers.Conv1D(filters//4, kernel_size=1, strides=1, padding="same")(branch1)

    branch2 = layers.Conv1D(filters//4, kernel_size=1, strides=1, padding="same")(x)

    branch21 = layers.Conv1D(filters//4, kernel_size=3, strides=1, padding="same")(branch2)
    branch22 = layers.Conv1D(filters//4, kernel_size=5, strides=1, padding="same")(branch2)
    branch23 = layers.Conv1D(filters//4, kernel_size=7, strides=1, padding="same")(branch2)

    out = layers.Concatenate()([branch1, branch21, branch22, branch23])
    out = layers.ReLU()(out)
    return out


input_layer = layers.Input(shape=(320, 1))

mod = layers.Conv1D(filters=8, kernel_size=7, strides=2, padding="same", activation="relu")(input_layer)

incep = inception_block(mod, filters=8)
incep = inception_block(incep, filters=8)
mod = layers.Add()([mod,incep])

mod = layers.Conv1D(filters=16, kernel_size=5, strides=2, padding="same", activation="relu")(mod)

incep = inception_block(mod, filters=16)
incep = inception_block(incep, filters=16)

mod = layers.Add()([mod,incep])

mod = layers.Conv1D(filters=32, kernel_size=3, strides=2, padding="same", activation="relu")(mod)

incep = inception_block(mod, filters=32)
incep = inception_block(incep, filters=32)

mod = layers.Add()([mod,incep])

mod = layers.GlobalAveragePooling1D()(mod)

output_layer = layers.Dense(5, activation="softmax")(mod)

model = models.Model(inputs=input_layer, outputs=output_layer)
model.summary()