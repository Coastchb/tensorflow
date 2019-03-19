#include "tensorflow/Tacotron2/include/preprocess.h"

namespace dream {
void get_sentences(string& text, vector<string> *sentences) {
    for (auto eos_symbol : EOS_SYMBOLS) {
        LOG(INFO) << eos_symbol;
        SplitStringToVector(text, eos_symbol.c_str(), true, sentences);
    }
}

}