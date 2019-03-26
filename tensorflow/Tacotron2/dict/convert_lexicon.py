import codecs
import sys
import os

_pad        = '_'
_eos        = '~'
sil         = 'SIL'
man_phones = ["a","ai","an","ang","ao","b","c","ch","d","e","ei",
              "en","eng","er","f","g","h","i","ia","ian","iang",
              "iao","ie","in","ing","iong","iu","j","k","l","m",
              "n","o","ong","ou","p","q","r","s","sh","t","u",
              "ua","uai","uan","uang","ui","un","uo","v","van",
              "ve","vn","w","x","xr","y","z","zh"]

eng_phones = ["AA", "AE", "AH", "AO",
              "AW", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY",
              "F", "G", "HH", "IH", "IY", "JH", "K", "L", "M",
              "N", "NG", "OW", "OY", "P", "R", "S", "SH", "T",
              "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH"]
tones = [str(x) for x in range(1,10)]
punc = [ "`", "^", ",", "."]

units = [_pad, _eos, sil] + man_phones + eng_phones + tones + punc
unit2id = {p: i for i, p in enumerate(units)}
id2unit = {i:p for i, p in enumerate(units)}

def main():
    src_lexicon = sys.argv[1]
    dst_lexicon = sys.argv[2]

    dst = []
    with codecs.open(src_lexicon,'r',encoding="utf-8") as fr:
        for line in fr.readlines():
            tokens = line.split()
            word = tokens[0]
            pron_ids = []
            for token in tokens[1:]:
                if(token[-1] >= '0' and token[-1] <= '9'):
                    pron_ids.append(unit2id[token[:-1]])
                    pron_ids.append(unit2id[token[-1]])
                else:
                    pron_ids.append(unit2id[token])
            dst.append("%s %s" % (word,' '.join([str(x) for x in pron_ids])))
    with codecs.open(dst_lexicon, 'w', encoding="utf-8") as fw:
        fw.writelines("\n".join(dst))
main()


