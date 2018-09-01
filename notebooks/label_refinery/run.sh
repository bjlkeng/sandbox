export CMDLINE=1
# PREFIX="label_refinery"
PREFIX="label_refinery-cifar10"

export OUTDIR="output-$(date +'%s')-$PREFIX"
mkdir -p $OUTDIR

for run_num in 1 2 3 4 5; do
    for use_preprocessing in 1 0; do
        echo "USE_PREPROCESSING=$use_preprocessing, RUN=$run_num"
        export USE_PREPROCESSING="$use_preprocessing"
        export RUN_NUM="$run_num"
        cp $PREFIX{,-use_pre${use_preprocessing}-run_num${RUN_NUM}}.ipynb
        jupyter nbconvert --to notebook --execute ${PREFIX}-use_pre${use_preprocessing}-run_num${RUN_NUM}.ipynb --ExecutePreprocessor.timeout=-1
        rm ${PREFIX}-use_pre${use_preprocessing}-run_num${RUN_NUM}.ipynb
        mv ${PREFIX}-use_pre${use_preprocessing}-run_num${RUN_NUM}.nbconvert.ipynb $OUTDIR
        mv results* $OUTDIR
    done
done
mv allresults.csv $OUTDIR
