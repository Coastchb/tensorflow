load("//tensorflow:tensorflow.bzl", "tf_cc_binary")


#tf_cc_binary(
#    name = "infer",
#    srcs = [
#        "infer.cc",
#    ],
#    deps = [
#        "//tensorflow/core:core_cpu",
#        "//tensorflow/core:framework",
#       "//tensorflow/core:framework_internal",
#        "//tensorflow/core:lib",
#        "//tensorflow/core:lib_internal",
#        "//tensorflow/core:protos_all_cc",
#        "//tensorflow/core:tensorflow",
#    ],
#)

cc_library(
	name = "preprocess",
	srcs = ["preprocess.cc"],
	hdrs = [
	    "include/preprocess.h",
	    "include/utils.h",
	    "include/common.h",
	],
	deps = [
         "//tensorflow/core:lib",
         "//tensorflow/core:lib_internal",
	]
)

tf_cc_binary(
    name = "preprocess_test",
    srcs = [
        "preprocess_test.cc",
    ],
    deps = [
        "//tensorflow/core:core_cpu",
        ":preprocess",
    ],
)

