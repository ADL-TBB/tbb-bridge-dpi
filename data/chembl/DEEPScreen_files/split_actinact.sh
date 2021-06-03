# Removes the first drug of every line as it is the one that is appended with _ACT / _INACT
# Then replaces all commas by newline to split every drug we want to use for querying on its own line
< chembl27_preprocessed_filtered_act_inact_comps_10.0_20.0_blast_comp_0.2.txt cut -f 2- -d ',' | tr ',' '\n' > chembl27_preprocessed_filtered_act_inact_comps_10.0_20.0_blast_comp_0.2_split.txt
