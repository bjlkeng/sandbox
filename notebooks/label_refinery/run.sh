export CMDLINE=1

export OUTDIR="output-$(date +'%s')"
mkdir -p $OUTDIR

for use_preprocessing in 0 1; do
    for run_num in 1 2 3 4 5; do
        echo "USE_PREPROCESSING=$use_preprocessing, RUN=$run_num"
        export USE_PREPROCESSING="$use_preprocessing"
        export RUN_NUM="$run_num"
        cp label_refinery{,-use_pre${use_preprocessing}-run_num${RUN_NUM}}.ipynb
        jupyter nbconvert --to notebook --execute label_refinery-use_pre${use_preprocessing}-run_num${RUN_NUM}.ipynb --ExecutePreprocessor.timeout=-1
        rm label_refinery-use_pre${use_preprocessing}-run_num${RUN_NUM}.ipynb
        mv label_refinery-use_pre${use_preprocessing}-run_num${RUN_NUM}.nbconvert.ipynb $OUTDIR
        mv results* $OUTDIR
    done
done
mv allresults.csv $OUTDIR
