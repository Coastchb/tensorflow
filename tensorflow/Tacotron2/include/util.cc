#include "tensorflow/Tacotron2/include/utils.h"

namespace dream{

void SplitStringToVector(const string& full, const char* delimiters,
                         bool omit_empty_strings,
                         vector<string>* out) {
    //CHECK(out != NULL);
    out->clear();

    size_t start = 0, end = full.size();
    size_t found = 0;
    while (found != string::npos) {
        found = full.find_first_of(delimiters, start);
        // start != end condition is for when the delimiter is at the end.
        if (!omit_empty_strings || (found != start && start != end))
            out->push_back(full.substr(start, found - start));
        start = found + 1;
    }
}

}
