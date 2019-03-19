#include "tensorflow/Tacotron2/include/preprocess.h"
#include "tensorflow/Tacotron2/include/common.h"
#include "tensorflow/core/platform/logging.h"

int main(int argc, char* argv[]) {
    string text = "据新华社消息？今天北京天气晴朗，微风?请大家适当户外活动！"; //"一眨眼，2019年的“金三银四”已经快过了一半。跟往年相比，今年找个好工作，要更多的去比拼“硬实力”。";
    vector<string> sentences;
    dream::get_sentences(text, &sentences);

    for(auto sentence: sentences) {
        LOG(INFO) << sentence;
    }

    return 0;
}