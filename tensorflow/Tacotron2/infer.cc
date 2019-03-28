#include <fstream>
#include <iomanip>
#include <unordered_set>
#include <vector>
#include <iostream>
#include <chrono>
#include <math.h>
#include <stdio.h>
#include <string>

#include "tensorflow/Tacotron2/include/lpcnet/arch.h"
#include "tensorflow/Tacotron2/include/lpcnet/lpcnet.h"
#include "tensorflow/Tacotron2/include/lpcnet/freq.h"
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

#include "tensorflow/Tacotron2/utils/preprocess.h"
#include "tensorflow/Tacotron2/utils/file_utils.h"
#include "tensorflow/Tacotron2/utils/string_utils.h"

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

template <typename Word>
bool write_word(std::ostream& outs, Word value, unsigned size = sizeof(Word))
{
    for (; size; --size, value >>= 8)
        outs.put(static_cast <char> (value & 0xFF));
    return true;
}

bool synthesize(int num_frames, float infeat_matrix[][ACT_DIM], std::string& wav_file) {
    LPCNetState *net;
    net = lpcnet_create();

    ofstream fw(wav_file, ios::binary );
    // Write the file headers
    fw << "RIFF----WAVEfmt ";     // (chunk size to be filled in later)
    write_word( fw,     16, 4 );  // no extension data
    write_word( fw,      1, 2 );  // PCM - integer samples
    write_word( fw,      1, 2 );  // two channels (stereo file)
    write_word( fw,  16000, 4 );  // samples per second (Hz)
    write_word( fw,  64000, 4 );  // (Sample Rate * BitsPerSample * Channels) / 8
    write_word( fw,      4, 2 );  // data block size (size of two integer samples, one for each channel, in bytes)
    write_word( fw,     16, 2 );  // number of bits per sample (use a multiple of 8)

    // Write the data chunk header
    size_t data_chunk_pos = fw.tellp();
    fw << "data----";  // (chunk size to be filled in later)

    for (int i = 0; i < num_frames; i++){

        float features[NB_FEATURES]; // 38-dim
        short pcm[FRAME_SIZE];
        memcpy(features, infeat_matrix[i], 18*sizeof(*features));
        memset(&features[18], 0, 18*sizeof(*features));
        memcpy(&features[36], &infeat_matrix[i][18], 2* sizeof(*features));

        lpcnet_synthesize(net, pcm, features, FRAME_SIZE);
        for(auto& x: pcm) {
            write_word(fw, x, 2);
        }
    }
    // (We'll need the final file size to fix the chunk sizes above)
    size_t file_length = fw.tellp();

    // Fix the data chunk header to contain the data size
    fw.seekp( data_chunk_pos + 4 );
    write_word( fw, file_length - data_chunk_pos + 8 );

    // Fix the file header to contain the proper RIFF chunk size, which is (file size - 8) bytes
    fw.seekp( 0 + 4 );
    write_word( fw, file_length - 8, 4 );
    fw.close();
    lpcnet_destroy(net);
    return true;
}

bool synthesize_sentence(const string& sentence, std::unique_ptr<tensorflow::Session>& sess,
                         map<string, vector<int>>& dict, int sentence_index) {
    std::vector<int> sentence_input;

    if (!explorer::preprocess(dict, sentence, &sentence_input)) {
        LOG(ERROR) << "error while preprocessing";
        return false;
    }

    std::cout << "final input for sentence:\"" << sentence << "\"\n";
    for(auto& x: sentence_input){
        std::cout << x << " ";
    }
    std::cout << std::endl;

    int input_len = sentence_input.size();
    std::vector<int32> text_length({input_len});
    std::vector<int32> split_infos({{input_len}});
    Tensor text_tensor(tensorflow::DT_INT32,
                       tensorflow::TensorShape({1, input_len}));
    Tensor length_tensor(tensorflow::DT_INT32,
                         tensorflow::TensorShape({1}));
    Tensor split_tensor(tensorflow::DT_INT32,
                        tensorflow::TensorShape({1,1}));


    std::copy(sentence_input.begin(), sentence_input.end(), text_tensor.flat<int32>().data());
    std::copy(text_length.begin(), text_length.end(), length_tensor.flat<int32>().data());
    std::copy(split_infos.begin(), split_infos.end(), split_tensor.flat<int32>().data());

    // Do inference.
    std::vector<Tensor> outputs;
    Status run_status = sess->Run({{"Tacotron_model/text:0", text_tensor},
                                      {"Tacotron_model/text_len:0", length_tensor},
                                      {"Tacotron_model/split_infos", split_tensor}},
                                     {"Tacotron_model/mel_outputs"}, {}, &outputs);

    if (!run_status.ok()) {
        LOG(ERROR) << "running model failed: " << run_status;
        return false;
    } else {
        LOG(INFO) << "congratulate! you make it!";
    }

    Tensor output_tensor = outputs[0];
    auto outputs_flat = output_tensor.flat<float>();
    int output_row = output_tensor.shape().dim_size(0);
    int output_dim = output_tensor.shape().dim_size(1);
    float acoutic_feat[output_row][ACT_DIM];
    if(output_dim != ACT_DIM) {
        LOG(ERROR) << "incompatible dimension:" << output_dim <<":" << ACT_DIM;
        return false;
    }

    for (int i = 0; i < output_row; i++) {
        std::copy_n(outputs_flat.data() + i * output_dim, output_dim, acoutic_feat[i]);
    }

    LOG(INFO) << "start to synthesize:";
    string pcm_file = "syn_wav/sen_" + std::to_string(sentence_index) + ".wav";
    if (!synthesize(output_row, acoutic_feat, pcm_file)){
        LOG(ERROR) << "error while synthesizing";
        return false;
    }
    return true;
}

int main(int argc, char* argv[]) {
    std::chrono::duration<double, std::milli> elapsed;
    auto time1 = std::chrono::high_resolution_clock::now();
    string graph = "";
    string input_file = "";
    bool verbose = false;
    std::vector<Flag> flag_list = {
        Flag("graph", &graph, "model to be executed"),
        Flag("file", &input_file, "input file containing text to be synthesized"),
        Flag("verbose", &verbose, "whether to log extra debugging information"),
    };
    string usage = tensorflow::Flags::Usage(argv[0], flag_list);
    const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
    if (!parse_result) {
        LOG(ERROR) << usage;
        return -1;
    }
    if (argc > 1) {
        LOG(ERROR) << "unknown argument " << argv[1] << "\n" << usage;
        return -1;
    }

    //ProfilerStart("test.prof");//开启性能分析

    // First we load and initialize the model.
    std::unique_ptr<tensorflow::Session> session;

    Status load_graph_status = LoadGraph(graph, &session);
    if (!load_graph_status.ok()) {
        LOG(ERROR) << load_graph_status;
        return -1;
    } else {
        LOG(INFO) << "load graph successfully!\n";
    }

    // load dict
    map<string, vector<int>> dict;
    if(!explorer::load_dict(explorer::dict_file, &dict)) {
        LOG(ERROR) << "error while loading dict";
        return -1;
    }

    // read the file
    std::vector<std::string> text;
    explorer::read_file(input_file, &text);

    int sentence_index = 0;

    for(auto& line: text) {
        // step 1: split text to sentences
        vector<string> sentences;
        if (!explorer::get_sentences(line, &sentences)) {
            cerr << "error while splitting text to sentences" << endl;
            return -1;
        }
        for (auto& sentence: sentences) {
            if(!synthesize_sentence(sentence, session, dict, sentence_index)) {
                LOG(ERROR) << "error while synthesizing the sentence.";
            }
            sentence_index++;
        }
    }
    //ProfilerStop();//停止性能分析
}

