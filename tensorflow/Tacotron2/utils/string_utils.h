#ifndef _STRING_UTILS_H
#define  _STRING_UTILS_H

#include "tensorflow/Tacotron2/utils/common.h"

namespace explorer{
    static string BOSON_KEY = "2DgGSC-8.33497.8yeNchBP6L9n";

    static string EOS_SYMBOLS[] = {"。", "！", "？", "?", "!"};
    static string INTONATION_LABLES[] = {"，", "；", "：", "——"};
    static string USELESS_LABLES[] = {"《", "》"};

    static string LAB_PROSODY_WORD = "`";
    static string LAB_PROSODY_PHRASE = "^";
    static string LAB_INTONATION_PHRASE = ",";
    static string LAB_SENTENCE = ".";
    static string LAB_EOS = "~";
    static string SIL = "SIL";

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
    bool split_text_to_sentence(const string& full,
                             bool omit_empty_strings,
                             vector<string>* out);
    bool replace_all(string&, const string, const string);
    bool exe_cmd(const char*,char*);
}

#endif
