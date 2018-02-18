export CMDLINE=1

export OUTDIR="output-$(date +'%s')"
mkdir -p $OUTDIR

for depth in 0 1 2 3; do
    echo "Resnet Depth=$depth"
    export RESNET_DEPTH=$depth
    cp vae-cifar10{,-depth$depth}.ipynb
    jupyter nbconvert --to notebook --execute vae-cifar10-depth$depth.ipynb --ExecutePreprocessor.timeout=-1
    rm vae-cifar10-depth$depth.ipynb
    mv vae-cifar10-depth${depth}.nbconvert.ipynb $OUTDIR
done
