import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

os.environ["GOOOOG"] = "1"
if os.environ["GOOOOG"]=="0":
    import tensorflow.keras as keras
else:
    import keras

import layers_brain as layers
import tensorflow as tf

long_path = "/DataCommon/ksoh/longitudinal/3class"
long_nc, long_ad = np.load(long_path + "/resized_quan_NC.npy"), np.load(long_path + "/resized_quan_AD.npy")
long_nc, long_ad = np.expand_dims(long_nc, axis=-1), np.expand_dims(long_ad, axis=-1)

ratio = 4
skip = True
class SonoNet:
    def __init__(self, ch=16):
        self.ch = ch
        tf.keras.backend.set_image_data_format("channels_last")
        self.enc_in_layer = layers.input_layer(input_shape=(96, 114, 96, 1), name="enc_in")
        self.build_model()

    def conv_bn_act(self, x, f, n, s=1, k=None, p="same", act=True, trans=False, out_p="auto"):
        if trans:
            c_layer = layers.conv_transpose
        else:
            c_layer = layers.conv

        if k:
            conv_l = c_layer(f=f, p=p, k=k, s=s, out_p=out_p, name=n+"_conv")
        else:
            conv_l = c_layer(f=f, p=p, name=n + "_conv")

        out = conv_l(x)
        norm_l = layers.batch_norm(name=n + "_norm")
        out = norm_l(out)

        if act:
            act_l = layers.relu(name=n + "_relu")
            out = act_l(out)
        return out

    def concat(self, x, y, n):
        concat_l = layers.concat(name=n + "_concat")
        return concat_l([x, y])

    def flatten_layer(self, x, n=None):
        flatten_l = layers.flatten(n + "_flatten")(x)
        return flatten_l

    def dense_layer(self, x, f, act=None, n=None):
        dense_l = layers.dense(f, act=None, name=n + "_dense")
        out = dense_l(x)

        if act:
            act_l = layers.relu(n + "_relu")
            out = act_l(out)
        return out

    def GALA_block(self, x, ch, ratio, n=None, SKIP=None):
        shortcut = x
        ch_reduced = ch // ratio

        # Receptive field
        recep1 = self.conv_bn_act(x=x, k=4, s=2, f=ch, p="same", act=True, n=n+"recep1")
        recep2 = layers.upsample(name="recep2")(recep1)
        if x.shape[2] != recep2.shape[2]:
            recep2 = self.conv_bn_act(x=recep2, k=(1, 2, 1), s=1, p="valid", f=ch, n=n+"reshape")

        # Local-attention mechanism (SE_module)
        squeeze = layers.global_avgpool(rank=3, name=n+"_GAP")(x)
        excitation = layers.dense(f=ch_reduced, act="relu", b=False, name=n+"_L_excitation1")(squeeze)
        L_out = layers.dense(f=ch, act=None, b=False, name=n+"_L_excitation2")(excitation)
        L_out = keras.layers.Reshape((-1, 1, 1, ch), name=n+"_L_reshape")(L_out)

        # Global-attention mechanism
        shrink = layers.conv(f=ch_reduced, k=1, s=1, p="same", act="relu", name=n+"_G_shrink")(x)
        G_out = layers.conv(f=1, k=1, s=1, p="same", act=None, name=n+"_G_collapse")(shrink)

        # Aggregation
        mul1 = keras.layers.Multiply()([recep2, L_out])
        mul2 = keras.layers.Multiply()([recep2, G_out])
        add = keras.layers.Add()([mul1, mul2])
        A = layers.tanh(add, name=n+"_tanh")

        activity_map = tf.norm(A, 'euclidean', axis=-1, name=n+"_A_norm")

        # Output
        x = keras.layers.Multiply()([shortcut, A])
        if SKIP: return layers.relu(n+"_relu")(keras.layers.Add()([x, shortcut])), activity_map
        else: return layers.relu(n+"_relu")(x), activity_map

    def build_model(self):
        enc_conv1_1 = self.conv_bn_act(x=self.enc_in_layer, f=self.ch, n="enc_conv1_1")
        enc_conv1_2 = self.conv_bn_act(x=enc_conv1_1, f=self.ch, n="enc_conv1_2")
        enc_conv1_GALA, map1 = self.GALA_block(x=enc_conv1_2, ch=self.ch, ratio=ratio, n="enc_conv1_GALA", SKIP=skip)
        enc_pool1 = layers.maxpool(name="enc_pool1")(enc_conv1_GALA)

        enc_conv2_1 = self.conv_bn_act(x=enc_pool1, f=self.ch * 2, n="enc_conv2_1")
        enc_conv2_2 = self.conv_bn_act(x=enc_conv2_1, f=self.ch * 2, n="enc_conv2_2")
        enc_conv2_GALA, map2 = self.GALA_block(x=enc_conv2_2, ch=self.ch * 2, ratio=ratio, n="enc_conv2_GALA", SKIP=skip)
        enc_pool2 = layers.maxpool(name="enc_pool2")(enc_conv2_GALA)

        enc_conv3_1 = self.conv_bn_act(x=enc_pool2, f=self.ch * 4, n="enc_conv3_1")
        enc_conv3_2 = self.conv_bn_act(x=enc_conv3_1, f=self.ch * 4, n="enc_conv3_2")
        enc_conv3_3 = self.conv_bn_act(x=enc_conv3_2, f=self.ch * 4, n="enc_conv3_3")
        enc_conv3_GALA, map3 = self.GALA_block(x=enc_conv3_3, ch=self.ch * 4, ratio=ratio, n="enc_conv3_GALA", SKIP=skip)
        enc_pool3 = layers.maxpool(name="enc_pool3")(enc_conv3_GALA)

        enc_conv4_1 = self.conv_bn_act(x=enc_pool3, f=self.ch * 8, n="enc_conv4_1")
        enc_conv4_2 = self.conv_bn_act(x=enc_conv4_1, f=self.ch * 8, n="enc_conv4_2")
        enc_conv4_3 = self.conv_bn_act(x=enc_conv4_2, f=self.ch * 8, n="enc_conv4_3")
        enc_conv4_GALA, map4 = self.GALA_block(x=enc_conv4_3, ch=self.ch * 8, ratio=ratio, n="enc_conv4_GALA", SKIP=skip)
        enc_pool4 = layers.maxpool(name="enc_pool4")(enc_conv4_GALA)

        enc_conv5_1 = self.conv_bn_act(x=enc_pool4, f=self.ch * 8, n="enc_conv5_1")
        enc_conv5_2 = self.conv_bn_act(x=enc_conv5_1, f=self.ch * 8, n="enc_conv5_2")
        enc_conv5_3 = self.conv_bn_act(x=enc_conv5_2, f=self.ch * 8, n="enc_conv5_3")
        enc_conv5_GALA, map5 = self.GALA_block(x=enc_conv5_3, ch=self.ch * 8, ratio=ratio, n="enc_conv5_GALA", SKIP=skip)

        enc_conv6_1 = self.conv_bn_act(x=enc_conv5_GALA, f=self.ch * 4, k=1, p="same", n="enc_conv6_1")
        enc_conv6_2 = tf.identity(self.conv_bn_act(x=enc_conv6_1, f=2, k=1, act=False, n="enc_conv6_2"))

        gap = layers.global_avgpool(rank=3, name="GAP")(enc_conv6_2)
        self.cls_out = layers.softmax(gap, name="softmax")

        self.cls_model = keras.Model({"cls_in": self.enc_in_layer}, {"cls_out": self.cls_out, "m1":map1, "m2":map2,
                                                                     "m3":map3, "m4":map4, "m5":map5}, name="cls_model")
        return self.cls_model


def get_model():
    import pickle
    model = SonoNet().cls_model
    # model.load_weights("/home/jsyoon/ks_innvestigate/3fold_cls_model_096/variables/variables")
    # weight_dict = {}
    # for a in model.variables:
    #     weight_dict[a.name] = a.numpy()
    # with open("/home/jsyoon/ks_innvestigate/brain_model_weights.pkl", "wb") as f:
    #     pickle.dump(weight_dict, f)
    # raise Exception

    with open("/home/jsyoon/ks_innvestigate/brain_model_weights.pkl", "rb") as f:
        weight_dict = pickle.load(f)

    new_weight_dict = {}
    for fff in weight_dict:
        old_f = fff
        if fff[0] == "b":
            if fff.startswith("batch_normalization/"):
                fff = fff.replace("batch_normalization/", "batch_normalization_1/")
            else:
                fff = "batch_normalization_%d/%s" % (
                    int(fff.split("_")[2].split("/")[0]) - 55, fff.split("/")[-1])
        elif fff[0] == "d":
            if fff.startswith("dense/"):
                fff = fff.replace("dense/", "dense_1/")
            else:
                fff = "dense_%d/%s" % (int(fff.split("_")[1].split("/")[0]), fff.split("/")[-1])

        if os.environ["GOOOOG"] == "0":
            new_weight_dict[old_f] = weight_dict[old_f]
        else:
            new_weight_dict[fff] = weight_dict[old_f]

    for l in model.layers:
        for old_w in l.weights:
            keras.backend.set_value(old_w, new_weight_dict[old_w.name])
    return model


model = get_model()
import innvestigate
from tqdm import tqdm

methods = [
    ("lrp.z", {}, "LRP-Z"),
    # ("deep_taylor", {}, "DeepTaylor"),
    ("deep_taylor.bounded", {"low": -0.5362483, "high": 2.397606}, "DeepTaylor"),
    ("guided_backprop", {}, "Guided Backprop")
]

all_res_AD = np.zeros(shape=(3, len(long_ad), 96, 114, 96, 1), dtype=np.float32)
all_res_NC = np.zeros(shape=(3, len(long_nc), 96, 114, 96, 1), dtype=np.float32)

for cnt1, method in enumerate(methods):
    ad_analyzer = innvestigate.create_analyzer(method[0], model, **method[1])
    nc_analyzer = innvestigate.create_analyzer(method[0], model, **method[1])
    all_res_AD[cnt1] = ad_analyzer.analyze(long_ad)
    all_res_NC[cnt1] = nc_analyzer.analyze(long_nc)

np.save("/home/jsyoon/ks_innvestigate/brain/lrp_AD.npy", all_res_AD[0])
np.save("/home/jsyoon/ks_innvestigate/brain/deeptaylor2_AD.npy", all_res_AD[1])
np.save("/home/jsyoon/ks_innvestigate/brain/guided_AD.npy", all_res_AD[2])

np.save("/home/jsyoon/ks_innvestigate/brain/lrp_NC.npy", all_res_NC[0])
np.save("/home/jsyoon/ks_innvestigate/brain/deeptaylor2_NC.npy", all_res_NC[1])
np.save("/home/jsyoon/ks_innvestigate/brain/guided_NC.npy", all_res_NC[2])
