#ifndef _PREPROCESS_H
#define _PREPROCESS_H

#include "tensorflow/Tacotron2/utils/string_utils.h"
//#include "tensorflow/core/platform/logging.h"

namespace explorer {
    static string dict_file = "tensorflow/Tacotron2/dict/lexicon.txt";
    static const string prosody_model_file = "models/prosody_1.model";

    bool get_sentences(string&, vector<string>*);
    bool segment_pos(const string&, char []);
    bool extract_pos(char [], vector<string>*);
    bool gen_prosody_feat(vector<string>*, vector<string>*);
    bool load_dict(string&, map<string,vector<int>>*);
    bool is_intonation_label(string&);
    bool gen_final_input(vector<string>*, map<string, vector<int>>*, vector<int>*);
    bool preprocess(map<string, vector<int>>&, const string&, vector<int>*);
}


#endif