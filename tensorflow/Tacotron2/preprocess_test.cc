#include "tensorflow/Tacotron2/utils/preprocess.h"
#include "tensorflow/Tacotron2/include/crfpp/crfpp.h"

int main(int argc, char* argv[]) {
    string text = "大家好good morning!据新华社消息？今天北京天气晴朗，微风?请大家适当户外活动"; //"一眨眼，2019年的“金三银四”已经快过了一半。跟往年相比，今年找个好工作，要更多的去比拼“硬实力”。";
    const string model_file = "models/prosody_1.model";

    // step 1: split text to sentences
    vector<string> sentences;
    explorer::get_sentences(text, &sentences);

    /*
    cout << "splited sentences:" << endl;
    for(auto sentence: sentences) {
       cout << sentence << endl;
    }*/

    // load dict
    map<string, vector<int>> dict;
    //cout << "load dict" << endl;
    explorer::load_dict(explorer::dict_file, &dict);
    //cout << "dict loaded" << endl;
    for(auto sentence: sentences) {
        // step 2: segment and POS
        char str_ret[2048*4] = "";
        vector<string> pos_ret;
        explorer::segment_pos(sentence, str_ret);
        explorer::extract_pos(str_ret, &pos_ret);

        /*
        cout << "segment and POS:\n";
        for(auto r: pos_ret) {
            cout << r << endl;
        }
         */

        // prepare feat for prosody prediction
        vector<string> feat_vec;
        explorer::gen_prosody_feat(&pos_ret, &feat_vec);

        /*
        cout << "input for prosody prediction:\n";
        for(auto f: feat_vec)
            cout << f << endl;
        */

        // step 3: predict prosody
        vector<string> prosody_output_vec;
        crfpp_test1(model_file, &feat_vec, &prosody_output_vec);
        /*
        cout << "predicted prosody:\n";
        for (auto s: prosody_output_vec) {
            cout << s << endl;
        }
         */

        // step 4: convert words to pronunciations (phone ids)
        //         and merge prosody
        vector<int> id_input;
        explorer::gen_final_input(&prosody_output_vec, &dict, &id_input);
        cout << "final id inputs:\n";
        for(auto i: id_input)
            cout << i << "\t";
        cout << endl;

    }

    return 0;
}