BATCH_SIZE=32
TEXT_LENGTH=256

MECAB_DICT_DIR=`mecab-config --dicdir`
MECAB_OPT=""
if [ -x ${MECAB_DICT_DIR}/mecab-ipadic-neologd ]; then
    MECAB_OPT="--mecab_dict ${MECAB_DICT_DIR}/mecab-ipadic-neologd"
fi

MODEL_DIR=./models/Japanese_L-12_H-768_A-12_E-30_BPE
CONF_FILE=${MODEL_DIR}/bert_config.json
VOCAB_FILE=${MODEL_DIR}/vocab.txt

TRAINED_MODEL=./results/masuda/net_trained_2000.pth
TSV_FILE=./data/masuda/train_2000.tsv

INDEX=0
HTML_FILE=./results/masuda/attention/att_2000_${INDEX}.html


function run_once() {
    poetry run python check_attention.py --batch_size ${BATCH_SIZE} --text_length ${TEXT_LENGTH} --index ${INDEX} --save_html ${HTML_FILE}  ${MECAB_OPT}   ${CONF_FILE}  ${TRAINED_MODEL}  ${TSV_FILE}  ${VOCAB_FILE}
}

function run_all() {
    for i in `seq 2000`
    do
    poetry run python check_attention.py --batch_size ${BATCH_SIZE} --text_length ${TEXT_LENGTH} --index $i --save_tsv ./results/masuda/attention/att_2000_$i  ${MECAB_OPT}   ${CONF_FILE}  ${TRAINED_MODEL}  ${TSV_FILE}  ${VOCAB_FILE}
    done
}

# run_once
run_all
