APPS_EVAL_DIR="~/apps/eval/"
SKIP_AMT=20
SAVE_LOC="~/apps/results"
MODEL_LOC='~/apps/checkpoints/final/'
TEST_LOC="~/apps/data_split/test.json"
TOTAL_PROBLEMS=10640

for (( i=0; i <= $TOTAL_PROBLEMS ; i+=$SKIP_AMT)) ; 
do 
    echo "$frac $i"
    jid1=$(sbatch --parsable start_slurm_gen.sbatch $APPS_EVAL_DIR $i $(($i+$SKIP_AMT)) $SAVE_LOC  $MODEL_LOC  $TEST_LOC )
    jid2=$(sbatch --dependency=afterany:$jid1 test_all_sols.sbatch $APPS_EVAL_DIR $i $(($i+$SKIP_AMT)) $SAVE_LOC  $TEST_LOC )
done

