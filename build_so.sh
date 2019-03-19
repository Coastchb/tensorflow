bazel build -c opt --copt=-mavx2 --copt=-mavx --copt=-mfma --copt=-mfpmath=both --copt=-msse4.2 -k //tensorflow:libtensorflow_cc.so
