"""Heavy ranker: DCN v2 in TensorFlow/Keras.

Sparse ids go through embeddings, counter features through log1p and a learned
normalisation layer, and the two-tower similarity rides along as one more dense
input. A 3-layer cross network (explicit feature crosses, x_{l+1} = x0 * (W x_l + b) + x_l)
runs in parallel with a deep MLP, as in DCN v2.
"""
from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
import keras

CAT = ["user", "video", "author", "tag", "dur", "active", "hour"]


@keras.saving.register_keras_serializable(package="reelfunnel")
class Cross(keras.layers.Layer):
    """One DCN v2 cross layer with a full-rank weight matrix."""

    def build(self, shape):
        d = shape[-1]
        self.w = self.add_weight(name="w", shape=(d, d), initializer="glorot_uniform")
        self.b = self.add_weight(name="b", shape=(d,), initializer="zeros")

    def call(self, x0, x):
        return x0 * (tf.matmul(x, self.w) + self.b) + x


def build_ranker(cardinality: dict[str, int], n_dense: int, dense_sample: np.ndarray,
                 emb_dim: int = 16, cross_layers: int = 3, deep=(256, 128), lr: float = 1e-3) -> keras.Model:
    inputs, parts = {}, []
    for name in CAT:
        inp = keras.Input(shape=(), dtype="int32", name=name)
        inputs[name] = inp
        parts.append(keras.layers.Embedding(cardinality[name], emb_dim, name=f"emb_{name}")(inp))
    dense_in = keras.Input(shape=(n_dense,), dtype="float32", name="dense")
    inputs["dense"] = dense_in
    norm = keras.layers.Normalization(name="dense_norm")
    norm.adapt(dense_sample)
    parts.append(norm(dense_in))

    x0 = keras.layers.Concatenate(name="x0")(parts)
    x = x0
    for i in range(cross_layers):
        x = Cross(name=f"cross_{i}")(x0, x)
    h = x0
    for i, units in enumerate(deep):
        h = keras.layers.Dense(units, activation="relu", name=f"deep_{i}")(h)
        h = keras.layers.Dropout(0.1)(h)
    out = keras.layers.Dense(1, activation="sigmoid", name="p_long_view")(keras.layers.Concatenate()([x, h]))
    model = keras.Model(inputs, out, name="dcn_v2_ranker")
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="binary_crossentropy",
                  metrics=[keras.metrics.AUC(name="auc")])
    return model


def dense_transform(x: np.ndarray, log_cols: np.ndarray) -> np.ndarray:
    """log1p on non-negative count columns, NaN to 0. Applied identically at train and serve time."""
    x = np.array(x, dtype=np.float32, copy=True)
    x[:, log_cols] = np.log1p(np.clip(x[:, log_cols], 0, None))
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def make_predictor(model: keras.Model, n_dense: int):
    """A traced inference function: avoids Keras' per-call Python overhead at serving time."""
    spec = {k: tf.TensorSpec((None,), tf.int32, name=k) for k in CAT}
    spec["dense"] = tf.TensorSpec((None, n_dense), tf.float32, name="dense")

    @tf.function(input_signature=[spec])
    def predict(batch):
        return tf.squeeze(model(batch, training=False), -1)

    return predict
