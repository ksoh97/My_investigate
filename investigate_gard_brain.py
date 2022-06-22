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
long_mci = np.load(long_path + "/resized_quan_MCI.npy")
long_mci = np.expand_dims(long_mci, axis=-1)

class SonoNet():
    def __init__(self, ch=16):
        self.ch = ch
        keras.backend.set_image_data_format("channels_last")
        self.enc_in_layer = layers.input_layer(name="enc_in")
        self.build_model()

    def conv_bn_act(self, x, f, n, s=1, k=None, p="same", act=True, trans=False, out_p="auto"):
        if trans:
            c_layer = layers.conv_transpose
        else:
            c_layer = layers.conv
        if k:
            conv_l = c_layer(f=f, p=p, k=k, s=s, out_p=out_p, name=n + "_conv")
        else:
            conv_l = c_layer(f=f, p=p, name=n + "_conv")
        out = conv_l(x)
        norm_l = layers.batch_norm(name=n + "_norm")
        out = norm_l(out)
        if act:
            act_l = layers.relu(name=n + "_relu")
            out = act_l(out)
        return out

    def flatten_layer(self, x, n=None):
        flatten_l = layers.flatten(n + "_flatten")(x)
        return flatten_l

    def dense_layer(self, x, f, act="relu", n=None):
        dense_l = layers.dense(f, act=act, name=n + "_dense")
        out = dense_l(x)
        return out

    def build_model(self):
        # Encoder
        enc_conv1_1 = self.conv_bn_act(x=self.enc_in_layer, f=self.ch, n="enc_conv1_1")
        enc_conv1_2 = self.conv_bn_act(x=enc_conv1_1, f=self.ch, n="enc_conv1_2")
        enc_pool1 = layers.maxpool(name="enc_pool1")(enc_conv1_2)

        enc_conv2_1 = self.conv_bn_act(x=enc_pool1, f=self.ch * 2, n="enc_conv2_1")
        enc_conv2_2 = self.conv_bn_act(x=enc_conv2_1, f=self.ch * 2, n="enc_conv2_2")
        enc_pool2 = layers.maxpool(name="enc_pool2")(enc_conv2_2)

        enc_conv3_1 = self.conv_bn_act(x=enc_pool2, f=self.ch * 4, n="enc_conv3_1")
        enc_conv3_2 = self.conv_bn_act(x=enc_conv3_1, f=self.ch * 4, n="enc_conv3_2")
        enc_conv3_3 = self.conv_bn_act(x=enc_conv3_2, f=self.ch * 4, n="enc_conv3_3")
        enc_pool3 = layers.maxpool(name="enc_pool3")(enc_conv3_3)

        enc_conv4_1 = self.conv_bn_act(x=enc_pool3, f=self.ch * 8, n="enc_conv4_1")
        enc_conv4_2 = self.conv_bn_act(x=enc_conv4_1, f=self.ch * 8, n="enc_conv4_2")
        enc_conv4_3 = self.conv_bn_act(x=enc_conv4_2, f=self.ch * 8, n="enc_conv4_3")
        enc_pool4 = layers.maxpool(name="enc_pool4")(enc_conv4_3)

        enc_conv5_1 = self.conv_bn_act(x=enc_pool4, f=self.ch * 8, n="enc_conv5_1")
        enc_conv5_2 = self.conv_bn_act(x=enc_conv5_1, f=self.ch * 8, n="enc_conv5_2")
        enc_conv5_3 = self.conv_bn_act(x=enc_conv5_2, f=self.ch * 8, n="enc_conv5_3")

        # Classifier
        flatten = self.flatten_layer(x=enc_conv5_3, n="flatten")
        dense = self.dense_layer(x=flatten, f=2, act=None, n="dense")

        self.cls_model = keras.Model(self.enc_in_layer, dense, name="cls_model")
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

    # AD vs. NC classifier's model
    # with open("/home/jsyoon/ks_innvestigate/AD_NC_brain_model_weights.pkl", "rb") as f: weight_dict = pickle.load(f)
    # new_weight_dict = {}
    # for fff in weight_dict:
    #     old_f = fff
    #     if fff[0] == "b":
    #         if fff.startswith("batch_normalization/"):
    #             fff = fff.replace("batch_normalization/", "batch_normalization_1/")
    #         else:
    #             fff = "batch_normalization_%d/%s" % (
    #                 int(fff.split("_")[2].split("/")[0]) - 55, fff.split("/")[-1])
    #     elif fff[0] == "d":
    #         if fff.startswith("dense/"):
    #             fff = fff.replace("dense/", "dense_1/")
    #         else:
    #             fff = "dense_%d/%s" % (int(fff.split("_")[1].split("/")[0]), fff.split("/")[-1])
    #
    #     if os.environ["GOOOOG"] == "0":
    #         new_weight_dict[old_f] = weight_dict[old_f]
    #     else:
    #         new_weight_dict[fff] = weight_dict[old_f]
    #
    # for l in model.layers:
    #     for old_w in l.weights:
    #         keras.backend.set_value(old_w, new_weight_dict[old_w.name])

    # MCI vs. AD classifier's model
    with open("/home/jsyoon/ks_innvestigate/MCI_AD_brain_model_weights.pkl", "rb") as f: weight_dict = pickle.load(f)
    new_weight_dict = {}
    for fff in weight_dict:
        old_f = fff
        if fff[0] == "b":
            if fff.startswith("batch_normalization/"):
                fff = fff.replace("batch_normalization/", "batch_normalization_1/")
            else:
                fff = "batch_normalization_%d/%s" % (
                    int(fff.split("_")[2].split("/")[0]) - 12, fff.split("/")[-1])
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

    # MCI vs. NC classifier's model
    # with open("/home/jsyoon/ks_innvestigate/NC_MCI_brain_model_weights.pkl", "rb") as f: weight_dict = pickle.load(f)
    # new_weight_dict = {}
    # for fff in weight_dict:
    #     old_f = fff
    #     if fff[0] == "b":
    #         if fff.startswith("batch_normalization/"):
    #             fff = fff.replace("batch_normalization/", "batch_normalization_1/")
    #         else:
    #             fff = "batch_normalization_%d/%s" % (
    #                 int(fff.split("_")[2].split("/")[0]) - 53, fff.split("/")[-1])
    #     elif fff[0] == "d":
    #         if fff.startswith("dense_3/"):
    #             fff = fff.replace("dense_3/", "dense_1/")
    #         else:
    #             fff = "dense_%d/%s" % (int(fff.split("_")[1].split("/")[0]), fff.split("/")[-1])
    #
    #     if os.environ["GOOOOG"] == "0":
    #         new_weight_dict[old_f] = weight_dict[old_f]
    #     else:
    #         new_weight_dict[fff] = weight_dict[old_f]
    #
    # for l in model.layers:
    #     for old_w in l.weights:
    #         keras.backend.set_value(old_w, new_weight_dict[old_w.name])

    return model


if os.environ["GOOOOG"] == "0":
    import tensorflow as tf
    from deepexplain.tensorflow import DeepExplain

    with tf.Session() as sess:
        with DeepExplain(session=sess) as de:
            model= get_model()

            X = model.input
            logits = model.output

            intgrad_AD = de.explain('intgrad', logits, X, dat_AD, ys=lbl_AD)
            deeplift_AD = de.explain('deeplift', logits, X, dat_AD, ys=lbl_AD)

            intgrad_NC = de.explain('intgrad', logits, X, dat_NC, ys=lbl_NC)
            deeplift_NC = de.explain('deeplift', logits, X, dat_NC, ys=lbl_NC)

            np.save("/home/jsyoon/temp/intgrad_AD.npy", intgrad_AD)
            np.save("/home/jsyoon/temp/deeplift_AD.npy", deeplift_AD)

            np.save("/home/jsyoon/temp/intgrad_NC.npy", intgrad_NC)
            np.save("/home/jsyoon/temp/deeplift_NC.npy", deeplift_NC)

else:
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
    all_res_MCI = np.zeros(shape=(3, len(long_mci), 96, 114, 96, 1), dtype=np.float32)
    # all_res_NC = np.zeros(shape=(3, len(long_nc), 96, 114, 96, 1), dtype=np.float32)

    for cnt1, method in enumerate(methods):
        ad_analyzer = innvestigate.create_analyzer(method[0], model, **method[1])
        mci_analyzer = innvestigate.create_analyzer(method[0], model, **method[1])
        # nc_analyzer = innvestigate.create_analyzer(method[0], model, **method[1])
        all_res_AD[cnt1] = ad_analyzer.analyze(long_ad)
        all_res_MCI[cnt1] = mci_analyzer.analyze(long_mci)
        # all_res_NC[cnt1] = nc_analyzer.analyze(long_nc)

    np.save("/home/jsyoon/ks_innvestigate/brain/lrp_AD.npy", all_res_AD[0])
    np.save("/home/jsyoon/ks_innvestigate/brain/deeptaylor2_AD.npy", all_res_AD[1])
    np.save("/home/jsyoon/ks_innvestigate/brain/guided_AD.npy", all_res_AD[2])

    np.save("/home/jsyoon/ks_innvestigate/brain/lrp_MCI.npy", all_res_MCI[0])
    np.save("/home/jsyoon/ks_innvestigate/brain/deeptaylor2_MCI.npy", all_res_MCI[1])
    np.save("/home/jsyoon/ks_innvestigate/brain/guided_MCI.npy", all_res_MCI[2])

    # np.save("/home/jsyoon/ks_innvestigate/brain/lrp_NC.npy", all_res_NC[0])
    # np.save("/home/jsyoon/ks_innvestigate/brain/deeptaylor2_NC.npy", all_res_NC[1])
    # np.save("/home/jsyoon/ks_innvestigate/brain/guided_NC.npy", all_res_NC[2])