#include "tensorflow/Tacotron2/utils/string_utils.h"

namespace explorer{

    bool split_text_to_vector_onece(const string& full,
                              const string& delimeter,
                              bool omit_empty_strings,
                              vector<string>* out) {
        out->clear();

        size_t start = 0, end = full.size();
        size_t delim_len = delimeter.size();
        size_t found = full.find(delimeter, start);

        if(found != string::npos){
            if(!omit_empty_strings || (found != start && (start != end)))
                out->push_back(full.substr(start, found - start));
            out->push_back(full.substr(found + delimeter.size()));
            return true;
        } else {
            return false;
        }

    }

    bool split_text_to_vector(const string& full,
                                const string& delimeter,
                                bool omit_empty_strings,
                                vector<string>* out) {
        out->clear();

        size_t start = 0, end = full.size();
        size_t delim_len = delimeter.size();
        size_t found = full.find(delimeter, start);
        while(start < end) {
            if(found == string::npos) {
                if(start != end)
                    out->push_back(full.substr(start, found - start));
                break;
            }
            if(!omit_empty_strings || (found != start && (start != end)))
                out->push_back(full.substr(start, found - start));
            start = found + delim_len;
            found = full.find(delimeter, start);
        }

        return true;
    }

    bool split_text_to_vector(const string& full,
                              const string& delimeter,
                              bool omit_empty_strings,
                              vector<int>* out) {
        out->clear();

        size_t start = 0, end = full.size();
        size_t delim_len = delimeter.size();
        size_t found = full.find(delimeter, start);
        try{
            while(start < end) {
                if(found == string::npos) {
                    if(start != end)
                        out->push_back(atoi(full.substr(start, found - start).c_str()));
                    break;
                }
                if(!omit_empty_strings || (found != start && (start != end)))
                    out->push_back(atoi(full.substr(start, found - start).c_str()));
                start = found + delim_len;
                found = full.find(delimeter, start);
            }
        }
        catch(exception& e){
                cout << e.what() << endl;
        }

        return true;
    }

    bool split_text_to_sentence(const string& full,
                             bool omit_empty_strings,
                             vector<string>* out) {
        out->clear();

        size_t start = 0, end = full.size();
        bool has_end = false;

        while(! has_end) {
            has_end = true;
            size_t found = end;
            string cur_delim = "";
            for(auto de: EOS_SYMBOLS) {
                size_t tmp_found = full.find(de, start);
                if(tmp_found != string::npos){
                    has_end = false;
                    if (found > tmp_found){
                        found = tmp_found;
                        cur_delim = de;
                    }
                }
            }

            // start != end condition is for when the delimiter is at the end.
            if (!omit_empty_strings || (found != start && start != end))
                out->push_back(full.substr(start, found - start));
            start = found + cur_delim.size();
        }
        return true;
    }

    bool exe_cmd(const char* cmd,char* result) {
        char buffer[256];                         //定义缓冲区
        FILE* pipe = popen(cmd, "r");            //打开管道，并执行命令
        if (!pipe)
            return false;                      //返回0表示运行失败

        while(!feof(pipe)) {
            if(fgets(buffer, 256, pipe)){             //将管道输出到result中
                //cout << "buff:" << buffer << endl;
                strcat(result,buffer);
            }
        }
        pclose(pipe);                            //关闭管道
        return true;                                 //返回1表示运行成功
    }

    bool replace_all(string& text, const string src_sym, const string tar_sym) {
        size_t src_sym_len = src_sym.size(), tar_sym_len = tar_sym.size();
        size_t found = text.find(src_sym, 0);
        while(found != string::npos) {
            text.replace(found, src_sym_len, tar_sym);
            found = text.find(src_sym, found + tar_sym_len);
        }
        return true;
    }


}
