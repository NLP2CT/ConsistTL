input_data=$1
teacher_data=$2
teacher_checkpoint=$3
output=$4

#back-translate
fairseq-interactive $teacher_data --path $teacher_checkpoint --input $input_data --batch-size 100 --buffer-size 101 --beam 5 --lenpen 1> $output.output
grep ^H $output.output \
| sed 's/^H\-//' \
| sort -n -k 1 \
| cut -f 3 > $output.de

fairseq-preprocess -s de --only-source --trainpref $output --destdir $output-bin --workers 1 --srcdict $teacher_data/dict.de.txt