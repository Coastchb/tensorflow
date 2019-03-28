#include "tensorflow/Tacotron2/utils/preprocess.h"
#include "tensorflow/Tacotron2/include/crfpp/crfpp.h"

namespace explorer {

    bool get_sentences(string& text, vector<string>* sentences) {
        return split_text_to_sentence(text, true, sentences);
    }

    bool segment_pos(const string& sentence, char result[]) {
        string command = "curl -X POST \\\n"
                         "     -H \"Content-Type: application/json\" \\\n"
                         "     -H \"Accept: application/json\" \\\n"
                         "     -H \"X-Token: " + BOSON_KEY + "\" \\\n"
                         "     --data \"\\\"" + sentence + "\\\"\" \\\n"
                         "     'http://api.bosonnlp.com/tag/analysis?space_mode=0&oov_level=3&t2s=0' 1>&1";
        if(! exe_cmd(command.c_str(), result)){
            return false;
        }

        return true;
    }

    bool extract_pos(char src_str[], vector<string>* ret) {
        smatch result;
        //regex pattern("\\w+\\^([A-Za-z]+-([A-Za-z]+)\\+[A-Za-z]+).*");
        regex pattern("\"([^,:]+)\"");

        string str = src_str;
        //cout << str << endl;

        //string* tmp_str = new string[1];
        vector<string> all_vec;
        int count = 0;
        while (regex_search(str, result, pattern))
        {
            all_vec.push_back(result[1]);
            count += 1;
            str = result.suffix().str();
        }

        if(count % 2 != 0){
            cout << "invalid count:" << count << endl;
            return false;
        }

        for (int i = 1; i < count/2; i++) {
            ret->push_back(all_vec.at(i) + "\t" + all_vec.at(i+count/2));
        }

        return true;
    }

    bool gen_prosody_feat(vector<string>* input, vector<string>* out) {
        out->clear();
        for(vector<string>::iterator it = input->begin();
                it != input->end(); it++) {
            vector<string> line_vector;
            split_text_to_vector(*it, "\t", true, &line_vector);
            if(line_vector.size() != 2) {
                cout << "invalid line:" << *it << endl;
                return false;
            }
            int word_len = int(line_vector.at(0).size() / 3);
            out->push_back(line_vector.at(0) + "\t" + to_string(word_len) + "\t" + line_vector.at(1));
        }
        return true;
    }

    // TODO 1: implement this function
    bool load_dict(string& filename, map<string,vector<int>>* dict_map) {
        ifstream fis(filename);

        if(fis.is_open()){
            char line[1024];
            vector<string> line_vec;
            vector<int> pron_vec;
            while(fis.getline(line, 1024)) {
                line_vec.clear();
                pron_vec.clear();
                split_text_to_vector_onece(line, " ", true, &line_vec);
                split_text_to_vector(line_vec.at(1), " ", true, &pron_vec);
                dict_map->emplace(line_vec.at(0), pron_vec);
            }
        } else {
            //LOG(ERROR) << "error while opening file:" << filename;
            cout << "error while opening file:" << filename << endl;
            return false;
        }
        return true;
    }

    bool is_intonation_label(string& w) {
        // TODO 2: evaluate the std::find() function on string array (DONE)
        string* array_begin_p = begin(INTONATION_LABLES);
        string* array_end_p = end(INTONATION_LABLES);
        string* p = find(array_begin_p, array_end_p, w);
        if ( p != array_end_p)
            return true;
        else
            return false;
    }

    bool gen_final_input(vector<string>* prosody_output, map<string, vector<int>>* dict, vector<int>* final_input) {
        // convert word to phone ids, with prosody merged
        cout << "****";
        for(auto& x: (*dict)[SIL])
            cout << x << endl;
        cout << "****";
        final_input->insert(final_input->end(), (*dict)[SIL].begin(), (*dict)[SIL].end());
        for (vector<string>::iterator it = prosody_output->begin();
                it != prosody_output->end(); it++) {

            vector<string> tokens;
            split_text_to_vector(*it, "\t", true, &tokens);
            string word = tokens.at(0), pause = tokens.at(3);

            // skip punctuations (， ; ：　...)
            if(is_intonation_label(word))
                continue;

            if(dict->find(word) == dict->end()) {
                //LOG(ERROR) << "OOV word:" << word;
                cout << "OOV word:" << word << endl;
                exit(1);
            }
            // TODO 3: evaluate insert() function (DONE)
            final_input->insert(final_input->end(), (*dict)[word].begin(), (*dict)[word].end());

            string prosody_label = "";

            if(pause == "1")
                prosody_label = LAB_PROSODY_WORD;
            else if(pause == "2")
                prosody_label = LAB_PROSODY_PHRASE;
            else if(pause == "3")
                prosody_label = LAB_INTONATION_PHRASE;
            else if(pause == "4")
                prosody_label = LAB_SENTENCE;
            else{
                //LOG(ERROR) << "unsupported pasue level:" << pause;
                cout << "unsupported pasue level:" << pause << endl;
                continue;
            }


            // TODO 4: look backward 1 word and determine current prosody (DONE)
            if( it != (prosody_output->end() - 1)) {
                if(is_intonation_label(*(it+1))){
                    if(pause < "3"){
                        //LOG(INFO) << "incorrect prediction prosody for " << word << " of pause level " << pause;
                        cout << "incorrect prediction prosody for " << word << " of pause level " << pause << endl;
                    }
                    prosody_label = LAB_INTONATION_PHRASE;
                }

            }

            final_input->insert(final_input->end(), (*dict)[prosody_label].begin(), (*dict)[prosody_label].end());

        }
        final_input->insert(final_input->end(), (*dict)[SIL].begin(), (*dict)[SIL].end());
        final_input->insert(final_input->end(), (*dict)[LAB_EOS].begin(), (*dict)[LAB_EOS].end());
        return true;
    }

    // preprocess a sentence
    bool preprocess(map<string, vector<int>>& dict, const string& sentence, vector<int>* input_ids) {
        // step 1: segment and POS
        char str_ret[2048 * 4] = "";
        vector <string> pos_ret;
        if(! (segment_pos(sentence, str_ret) && extract_pos(str_ret, &pos_ret))) {
            cerr << "error while getting POS" << endl;
            return false;
        }

        // step 2: prepare feat for prosody prediction
        vector <string> feat_vec;
        if(! gen_prosody_feat(&pos_ret, &feat_vec)) {
            cerr << "error while getting prosody feat" << endl;
            return false;
        }

        // step 3: predict prosody
        vector <string> prosody_output_vec;
        crfpp_test1(prosody_model_file, &feat_vec, &prosody_output_vec);

        // step 4: convert words to pronunciations (phone ids)
        //         and merge prosody
         if(!gen_final_input(&prosody_output_vec, &dict, input_ids)) {
             cerr << "error while getting final ids";
             return false;
         }
         return true;
    }
}