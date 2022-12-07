# install for fairseq and pytorch
cd ..
cd ConsisTL
pip install --editable .
cd ..
cd full_process_scripts

# train two parent model
## train for en-de
### path of binarized parent model training data
BIN_TEACHER_DATA=${BIN_TEACHER_DATA}
bash train_parent.sh en de $BIN_TEACHER_DATA
## train for de-en
bash train_parent.sh de en $BIN_TEACHER_DATA

#gen synthetic de-en for tr-en
## English sentences in child data
CHILD_EN=${CHILD_EN}
## path of trained reversed teacher checkpoint
REVERSED_TEACHER_CHECKPOINT=${REVERSED_TEACHER_CHECKPOINT}
## auxiliary source
AUX_SRC_BIN=${AUX_SRC_BIN}
bash gen.sh $CHILD_EN $BIN_TEACHER_DATA $REVERSED_TEACHER_CHECKPOINT $AUX_SRC_BIN

#switch checkpoint
## path of initialized checkpoint
INIT_CHECKPOINT=${INIT_CHECKPOINT}
## path of student data
BIN_STUDENT_DATA=${BIN_STUDENT_DATA}
## path of teacher checkpoint
python ../ConsisTL/preprocessing_scripts/TM.py --checkpoint $TEACHER_CHECKPOINT --output $INIT_CHECKPOINT --parent-dict $BIN_TEACHER_DATA/dict.de.txt --child-dict $BIN_STUDENT_DATA/dict.tr.txt --switch-dict src

#train for TM-TL
bash train.sh $BIN_STUDENT_DATA $INIT_CHECKPOINT

#train for ConsisTL
bash ConsisTL.sh $AUX_SRC_BIN $TEACHER_CHECKPOINT $BIN_TEACHER_DATA $BIN_STUDENT_DATA $INIT_CHECKPOINT
