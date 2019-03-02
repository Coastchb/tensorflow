#include <fstream>
#include <iomanip>
#include <unordered_set>
#include <vector>
#include <iostream>
#include <chrono>

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
    
    std::vector<int32> text_input({{0,1,2,3,4,5,6,7,8,9}});
    std::vector<int32> text_length({10});
    std::vector<int32> split_infos({{10}});
    Tensor text_tensor(tensorflow::DT_INT32,
                             tensorflow::TensorShape({1, 10}));
    Tensor length_tensor(tensorflow::DT_INT32,
                         tensorflow::TensorShape({1}));
    Tensor split_tensor(tensorflow::DT_INT32,
                        tensorflow::TensorShape({1,1}));

    /*
    LOG(INFO) << text_input.size();
    for(auto iter = text_input.begin(); iter != text_input.end(); iter++){
        LOG(INFO) << *iter;
    }*/

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
        for(auto iter=outputs.begin(); iter != outputs.end(); iter++) {
            LOG(INFO) << (*iter).flat<float>().data();
        }
        LOG(INFO) << "Congratulate! You make it!";
    }
    ProfilerStop();//停止性能分析

}
