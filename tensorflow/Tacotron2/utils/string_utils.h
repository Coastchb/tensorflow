#ifndef _UTILS_H
#define  _UTILS_H

#include "tensorflow/Tacotron2/utils/common.h"

namespace explorer{
    static string BOSON_KEY = "2DgGSC-8.33497.8yeNchBP6L9n";

    static string EOS_SYMBOLS[] = {"。", "！", "？", "?", "!"};
    static string INTONATION_LABLES[] = {"，", "；", "：", "——"};

    static string LAB_PROSODY_WORD = "`";
    static string LAB_PROSODY_PHRASE = "^";
    static string LAB_INTONATION_PHRASE = ",";
    static string LAB_EOS = ".";

    bool split_text_to_vector_onece(const string&text,
                              const string& delimeter,
                              bool omit_empty_strings,
                              vector<string>* out);
    bool split_text_to_vector(const string&text,
                              const string& delimeter,
                              bool omit_empty_strings,
                              vector<string>* out);
    bool split_text_to_vector(const string&text,
                              const string& delimeter,
                              bool omit_empty_strings,
                              vector<int>* out);
    void split_text_to_sentence(const string& full,
                             bool omit_empty_strings,
                             vector<string>* out);
    bool exe_cmd(const char*,char*);
}

#endif
