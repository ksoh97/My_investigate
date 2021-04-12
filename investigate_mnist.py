import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

os.environ["GOOOOG"] = "1"
if os.environ["GOOOOG"]=="0":
    import tensorflow.keras as keras
else:
    import keras
import layers_mnist as layers


class SmallNet:
    def __init__(self, ch=32):
        self.ch = ch
        self.layers = {}
        self.input = layers.input_layer(name="input")
        self.c1, self.c2, self.c3 = layers.c1(name="c1"), layers.c2(name="c2"), layers.c3(name="c3")
        self.smallnet()
    def conv_bn_act(self, x, f, n, s=1, k=None, p="same", out_p=None, act=True, trans=False):
        if trans:
            c_layer = layers.conv_transpose
        else:
            c_layer = layers.conv
        if k:
            conv_l = c_layer(f=f, p=p, k=k, s=s, out_p=out_p, name=n+"_conv")
        else:
            conv_l = c_layer(f=f, p=p, name=n+"_conv")
        out = conv_l(x)
        norm_l = layers.batch_norm(name=n+"_norm")
        out = norm_l(out)
        self.layers[n+"_conv"] = conv_l
        self.layers[n+"_norm"] = norm_l
        if act:
            act_l = layers.relu(name=n + "_relu")
            self.layers[n + "_relu"] = act_l
            out = act_l(out)
        return out
    def conv_bn_act_reuse(self, x, n, act=True):
        out = self.layers[n+"_conv"](x)
        out = self.layers[n+"_norm"](out)
        if act:
            out = self.layers[n + "_relu"](out)
        return out
    def concat(self, x, y, n):
        concat_l = layers.concat(name=n+"_concat")
        self.layers[n+"_concat"] = concat_l
        return concat_l([x, y])
    def flatten_layer(self, x, n=None):
        flatten_l = layers.flatten(n+"_flatten")(x)
        self.layers[n+"_flatten"] = flatten_l
        return flatten_l
    def flatten_layer_reuse(self, x, n):
        return self.layers[n+"_flatten"](x)
    def dense_layer(self, x, f, act="relu", n=None):
        dense_l = layers.dense(f, act=None, name=n + "_dense")
        out = dense_l(x)
        self.layers[n+"_dense"] = dense_l
        if act:
            act_l = layers.relu(n + "_relu")
            self.layers[n + "_relu"] = act_l
            out = act_l(out)
        return out
    def dense_layer_reuse(self, x, n, act=True):
        out = self.layers[n+"_dense"](x)
        if act: out = self.layers[n+"_relu"](out)
        return out
    def smallnet(self):
        """
            Do not change layers' names!!!
        """
        enc_conv1_1 = self.conv_bn_act(x=self.input, k=3, s=1, f=self.ch, p="same", n="enc_conv1_1")
        enc_conv1_2 = self.conv_bn_act(x=enc_conv1_1, k=4, s=2, f=self.ch, p="same", n="enc_conv1_2")
        enc_conv2_1 = self.conv_bn_act(x=enc_conv1_2, k=3, s=1, f=self.ch * 2, p="same", n="enc_conv2_1")
        enc_conv2_2 = self.conv_bn_act(x=enc_conv2_1, k=4, s=2, f=self.ch * 2, p="same", n="enc_conv2_2")
        enc_conv3_1 = self.conv_bn_act(x=enc_conv2_2, k=3, s=1, f=self.ch * 4, p="same", n="enc_conv3_1")
        enc_conv3_2 = self.conv_bn_act(x=enc_conv3_1, k=4, s=2, f=self.ch * 4, p="same", n="enc_conv3_2")
        enc_conv4_1 = self.conv_bn_act(x=enc_conv3_2, k=3, s=1, f=self.ch * 8, p="same", n="enc_conv4_1")
        enc_conv4_2 = self.conv_bn_act(x=enc_conv4_1, k=4, s=2, f=self.ch * 8, p="same", n="enc_conv4_2")
        fc1 = self.flatten_layer(x=enc_conv4_2, n="enc_flatten")
        drop5 = layers.dropout(rate=0.5, name="enc_drop5")(fc1)
        dense_1 = self.dense_layer(x=drop5, f=128, n="dense_1")
        drop6 = layers.dropout(rate=0.25, name="enc_drop6")(dense_1)
        dense_2 = self.dense_layer(x=drop6, f=10, act=None, n="dense_3")
        self.cls_model = keras.Model(self.input, dense_2, name="cls_model")
        return self.cls_model


def get_model():
    import pickle
    model = SmallNet().cls_model

    # model.load_weights("/home/jsyoon/data/ksoh/models/mnist/variables/variables")
    # weight_dict = {}
    # for a in model.variables:
    #     weight_dict[a.name] = a.numpy()
    # with open("/home/jsyoon/temp/mnist_model_weights.pkl", "wb") as f:
    #     pickle.dump(weight_dict, f)
    # raise Exception

    with open("/home/jsyoon/ks_innvestigate/mnist_model_weights.pkl", "rb") as f:
        weight_dict = pickle.load(f)

    new_weight_dict = {}
    for fff in weight_dict:
        old_f = fff
        if fff[0] == "b":
            if fff.startswith("batch_normalization/"):
                fff = fff.replace("batch_normalization/", "batch_normalization_1/")
            else:
                fff = "batch_normalization_%d/%s" % (
                    int(fff.split("_")[2].split("/")[0]) - 25, fff.split("/")[-1])
        elif fff[0] == "d":
            if fff.startswith("dense/"):
                fff = fff.replace("dense/", "dense_1/")
            else:
                fff = "dense_%d/%s" % (int(fff.split("_")[1].split("/")[0]) - 3, fff.split("/")[-1])

        if os.environ["GOOOOG"] == "0":
            new_weight_dict[old_f] = weight_dict[old_f]
        else:
            new_weight_dict[fff] = weight_dict[old_f]

    for l in model.layers:
        for old_w in l.weights:
            keras.backend.set_value(old_w, new_weight_dict[old_w.name])
    return model



from keras.datasets import mnist
def fetch_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(60000, 28, 28, 1)
    x_test = x_test.reshape(10000, 28, 28, 1)

    x_train = x_train.astype("float32")/255.
    x_test = x_test.astype("float32")/255.

    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = fetch_data()

y_test = np.eye(10)[y_test]

if os.environ["GOOOOG"] == "0":
    import tensorflow as tf
    from deepexplain.tensorflow import DeepExplain

    with tf.Session() as sess:
        with DeepExplain(session=sess) as de:
            model= get_model()
            X = model.input
            logits = model.output

            intgrad_MNIST = de.explain('intgrad', logits, X, x_test, ys=y_test)
            deeplift_MNIST = de.explain('deeplift', logits, X, x_test, ys=y_test)

            np.save("/home/jsyoon/temp/intgrad_MNIST.npy", intgrad_MNIST)
            np.save("/home/jsyoon/temp/deeplift_MNIST.npy", deeplift_MNIST)

else:
    model = get_model()
    import innvestigate
    from tqdm import tqdm

    methods = [
        # ("lrp.z", {}, "LRP-Z"),
        # ("deep_taylor.bounded", {"low": 0., "high": 1.}, "DeepTaylor"),
        # ("guided_backprop", {}, "Guided Backprop",)
        # ("deconvnet", {}, "Deconvnet"),
        # ("integrated_gradients", {"reference_inputs": 0}, "Integrated Gradients"),
        ("smoothgrad", {"noise_scale": 0.1, "postprocess": "square"}, "SmoothGrad")
    ]

    all_res = np.zeros(shape=(1, len(x_test), 28, 28, 1), dtype=np.float32)

    for cnt1, method in enumerate(methods):
        all_res[cnt1] = innvestigate.create_analyzer(method[0], model, **method[1]).analyze(x_test)

    # np.save("/home/jsyoon/ks_innvestigate/mnist/lrpZ_MNIST.npy", all_res[0])
    # np.save("/home/jsyoon/ks_innvestigate/mnist/deeptaylor_MNIST.npy", all_res[1])
    # np.save("/home/jsyoon/ks_innvestigate/mnist/guided_MNIST.npy", all_res[2])
    # np.save("/home/jsyoon/ks_innvestigate/mnist/deconvnet_MNIST.npy", all_res[0])
    # np.save("/home/jsyoon/ks_innvestigate/mnist/integrated_MNIST.npy", all_res[1])
    # np.save("/home/jsyoon/ks_innvestigate/mnist/smoothgrad_MNIST.npy", all_res[2])
