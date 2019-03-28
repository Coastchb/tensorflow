#include "tensorflow/Tacotron2/utils/file_utils.h"

namespace explorer{
    bool read_file(string& filename,vector<string>* ret){
        fstream fin(filename, ios_base::in);
        if (fin.is_open()){
            char line[1024];
            while(fin.getline(line, 1024)) {
                ret->push_back(string(line));
            }
        } else {
            cerr << "error while opening file " << filename << endl;
            return false;
        }

        return true;
    }
}