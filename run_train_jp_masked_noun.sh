#!/bin/bash
BATCH_SIZE=32
TEXT_LENGTH=300
EPOCH=5

MECAB_DICT_DIR=`mecab-config --dicdir`
MECAB_OPT=""
if [ -x ${MECAB_DICT_DIR}/mecab-ipadic-neologd ]; then
    MECAB_OPT="--mecab_dict ${MECAB_DICT_DIR}/mecab-ipadic-neologd"
fi

MODEL_DIR=./models/Japanese_L-12_H-768_A-12_E-30_BPE
CONF_FILE=${MODEL_DIR}/bert_config.json
BASE_MODEL=${MODEL_DIR}/pytorch_model.bin
VOCAB_FILE=${MODEL_DIR}/vocab.txt

TRAIN_TSV=./data/masuda/train_20201204.tsv
SAVE_PATH=./results/masuda/net_trained_20101204_masked_noun.pth
LOG_FILE=./results/masuda/train_20201204.log


function run_once() {
    poetry run python train.py --batch_size ${BATCH_SIZE} --text_length ${TEXT_LENGTH} --epoch ${EPOCH}  ${MECAB_OPT}  --save_path ${SAVE_PATH}  ${CONF_FILE}  ${BASE_MODEL}  ${TRAIN_TSV}  ${VOCAB_FILE} >& ${LOG_FILE} &
    tail -f ${LOG_FILE}
}

run_once
