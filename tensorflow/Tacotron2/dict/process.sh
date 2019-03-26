
# replace some incompatible phones
 cat src_lexicon.txt |sed "s:ee er5:xr5:g"  | \
    sed "s:ee ::g" |sed "s:aa ::g" |sed "s:ii:y:g" | \
    sed "s:i[xyz]:i:g"  |sed "s:oo ::g" | sed "s:ueng:eng:g" | \
    sed "s:uu:w:g" | sed "s:vv:y:g" > dst_lexicon.txt

# convert phones to ids
 python convert_lexicon.py dst_lexicon.txt lexicon.txt