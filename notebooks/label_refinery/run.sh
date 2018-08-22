export CMDLINE=1

export OUTDIR="output-$(date +'%s')"
mkdir -p $OUTDIR

for dropout in 0.0 0.1 0.2; do
    for run_num in 1 2 3 4 5; do
        echo "Dropout=$dropout, RUN=$run_num"
        export DROPOUT="$dropout"
        export RUN_NUM="$run_num"
        cp label_refinery{,-dropout${dropout}-run_num${RUN_NUM}}.ipynb
        jupyter nbconvert --to notebook --execute label_refinery-dropout${dropout}-run_num${RUN_NUM}.ipynb --ExecutePreprocessor.timeout=-1
        rm label_refinery-dropout${dropout}-run_num${RUN_NUM}.ipynb
        mv label_refinery-dropout${dropout}-run_num${RUN_NUM}.nbconvert.ipynb $OUTDIR
        mv results* $OUTDIR
    done
done
