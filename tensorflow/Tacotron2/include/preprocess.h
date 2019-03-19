#ifndef _PREPROCESS_H
#define _PREPROCESS_H

#include "tensorflow/Tacotron2/include/utils.h"
#include "tensorflow/core/platform/logging.h"

namespace dream {
void get_sentences(string& text, vector <string> *sentences);
}

#endif