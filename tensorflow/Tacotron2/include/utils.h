#ifndef _UTILS_H
#define  _UTILS_H

#include "tensorflow/Tacotron2/include/common.h"

namespace dream{
static string EOS_SYMBOLS[] = {"？"};

void SplitStringToVector(const std::string& full, const char* delimiters,
                         bool omit_empty_strings,
                         std::vector<std::string>* out);
}

#endif
