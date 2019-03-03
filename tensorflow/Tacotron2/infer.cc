#include <fstream>
#include <iomanip>
#include <unordered_set>
#include <vector>
#include <iostream>
#include <chrono>
#include <math.h>
#include <stdio.h>

#include "../LPCNet/src/arch.h"
#include "../LPCNet/src/lpcnet.h"
#include "../LPCNet/src/freq.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/wav/wav_io.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "gperftools/profiler.h"

// These are all common classes it's handy to reference with no namespace.
using tensorflow::Flag;
using tensorflow::Status;
using tensorflow::Tensor;
using tensorflow::int32;
using tensorflow::int64;
using tensorflow::string;
using tensorflow::uint16;
using tensorflow::uint32;

const int ACT_DIM = 20;

// Reads a model graph definition from disk, and creates a session object you
// can use to run it.
Status LoadGraph(const string& graph_file_name,
                 std::unique_ptr<tensorflow::Session>* session) {
    tensorflow::GraphDef graph_def;
    Status load_graph_status =
            ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
    if (!load_graph_status.ok()) {
        return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                            graph_file_name, "'");
    }

    tensorflow::SessionOptions options;
    options.config.set_allow_soft_placement(true);

    session->reset(tensorflow::NewSession(options));
    Status session_create_status = (*session)->Create(graph_def);
    if (!session_create_status.ok()) {
        return session_create_status;
    }
    return Status::OK();
}


int synthesize(int num_frames, float infeat_matrix[][ACT_DIM], std::string out_file="out.pcm") {
    auto syn_start = std::chrono::high_resolution_clock::now();

    FILE *fout;
    LPCNetState *net;
    net = lpcnet_create();

    fout = fopen(out_file.c_str(), "wb");
    if (fout == NULL) {
        fprintf(stderr, "Can't open %s\n", out_file);
        exit(1);
    }

    std::chrono::duration<double, std::milli> elapsed;
    auto time1 = std::chrono::high_resolution_clock::now();
    elapsed = time1 - syn_start;
    std::cout << "syn1 consumed: " << elapsed.count() << "ms" << std::endl;

    for (int i = 0; i < num_frames; i++){

        float features[NB_FEATURES]; // 38-dim
        short pcm[FRAME_SIZE];
        memcpy(features, infeat_matrix[i], 18*sizeof(*features));
        memset(&features[18], 0, 18*sizeof(*features));
        memcpy(&features[36], &infeat_matrix[i][18], 2* sizeof(*features));

        lpcnet_synthesize(net, pcm, features, FRAME_SIZE);
        fwrite(pcm, sizeof(pcm[0]), FRAME_SIZE, fout);

    }
    auto time2 = std::chrono::high_resolution_clock::now();
    elapsed = time2 - time1;
    std::cout << "syn2 consumed: " << elapsed.count() << "ms" << std::endl;
    fclose(fout);
    lpcnet_destroy(net);
}

int main(int argc, char* argv[]) {
    std::chrono::duration<double, std::milli> elapsed;
    auto time1 = std::chrono::high_resolution_clock::now();
    string graph = "";
    bool verbose = false;
    std::vector<Flag> flag_list = {
        Flag("graph", &graph, "model to be executed"),
        Flag("verbose", &verbose, "whether to log extra debugging information"),
    };
    string usage = tensorflow::Flags::Usage(argv[0], flag_list);
    const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
    if (!parse_result) {
        LOG(ERROR) << usage;
        return -1;
    }
    if (argc > 1) {
        LOG(ERROR) << "Unknown argument " << argv[1] << "\n" << usage;
        return -1;
    }

    std::cout << graph << std::endl;

    ProfilerStart("test.prof");//开启性能分析

    // First we load and initialize the model.
    std::unique_ptr<tensorflow::Session> session;

    auto time2 = std::chrono::high_resolution_clock::now();
    elapsed = time2 - time1;
    std::cout << "preprocess consumed: " << elapsed.count() << "ms" << std::endl;

    Status load_graph_status = LoadGraph(graph, &session);
    if (!load_graph_status.ok()) {
        LOG(ERROR) << load_graph_status;
        return -1;
    } else {
        LOG(INFO) << "Load graph successfully!\n";
    }

    auto time3 = std::chrono::high_resolution_clock::now();
    elapsed = time3 - time2;
    std::cout << "load graph consumed: " << elapsed.count() << "ms" << std::endl;
    
    std::vector<int32> text_input({{35, 32, 39, 39, 42, 59, 24, 42, 45, 39, 31, 54, 35, 32, 39, 39, 42,
                                     59,  4, 42, 28, 46, 47, 61,  4, 42, 41, 34, 45, 28, 47, 48, 39, 28,
                                     47, 36, 42, 41, 46, 54,  1}});
    std::vector<int32> text_length({41});
    std::vector<int32> split_infos({{41}});
    Tensor text_tensor(tensorflow::DT_INT32,
                             tensorflow::TensorShape({1, 41}));
    Tensor length_tensor(tensorflow::DT_INT32,
                         tensorflow::TensorShape({1}));
    Tensor split_tensor(tensorflow::DT_INT32,
                        tensorflow::TensorShape({1,1}));


    std::copy(text_input.begin(), text_input.end(), text_tensor.flat<int32>().data());
    std::copy(text_length.begin(), text_length.end(), length_tensor.flat<int32>().data());
    std::copy(split_infos.begin(), split_infos.end(), split_tensor.flat<int32>().data());

    auto time4 = std::chrono::high_resolution_clock::now();
    elapsed = time4 - time3;
    std::cout << "prepare data consumed: " << elapsed.count() << "ms" << std::endl;

    // Do inference.
    std::vector<Tensor> outputs;
    Status run_status = session->Run({{"Tacotron_model/text:0", text_tensor},
                                      {"Tacotron_model/text_len:0", length_tensor},
                                      {"Tacotron_model/split_infos", split_tensor}},
                                     {"Tacotron_model/mel_outputs"}, {}, &outputs);
    auto time5 = std::chrono::high_resolution_clock::now();
    elapsed = time5 - time4;
    std::cout << "inference consumed: " << elapsed.count() << "ms" << std::endl;

    if (!run_status.ok()) {
        LOG(ERROR) << "Running model failed: " << run_status;
        return -1;
    } else {
        LOG(INFO) << "Congratulate! You make it!";
    }

    Tensor output_tensor = outputs[0];
    //std::vector<std::vector<float>> output_data;
    auto outputs_flat = output_tensor.flat<float>();
    int output_row = output_tensor.shape().dim_size(0);
    int output_dim = output_tensor.shape().dim_size(1);
    float acoutic_feat[output_row][ACT_DIM];
    if(output_dim != ACT_DIM) {
        LOG(INFO) << "incompatible dimension:" << output_dim <<":" << ACT_DIM;
        exit(1);
    }

    //auto outputs_matrix = output_tensor.matrix<float>();
    //std::cout << outputs_matrix;
    //LOG(INFO) << outputs_matrix(0);

    double mean[ACT_DIM];
    double var[ACT_DIM];
    FILE* fmean = fopen("mean","r");
    FILE* fvar = fopen("var", "r");
    fread(mean, sizeof(double), ACT_DIM, fmean);
    fread(var, sizeof(double), ACT_DIM, fvar);


    for (int i = 0; i < output_row; i++) {
        //std::vector<float> output_frame;
        //float array_frame[output_dim];
        //output_frame.reserve(output_dim);
        //std::copy_n(outputs_flat.data() + i * output_dim, output_dim,  // outputs_flat.data() to be removed
        //            std::back_inserter(output_frame));

        std::copy_n(outputs_flat.data() + i * output_dim, output_dim, acoutic_feat[i]);

        for(int j = 0; j < ACT_DIM; j++) {
            acoutic_feat[i][j] = acoutic_feat[i][j] * var[j] + mean[j];
        }
        //output_data.emplace_back(std::move(output_frame));
    }

    LOG(INFO) << "start to synthesize:";
    synthesize(output_row, acoutic_feat);
    ProfilerStop();//停止性能分析

}

