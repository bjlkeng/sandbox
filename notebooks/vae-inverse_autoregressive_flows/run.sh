export CMDLINE=1
jupyter nbconvert --to notebook --execute vae-iaf-mnist.ipynb --ExecutePreprocessor.timeout=-1
jupyter nbconvert --to notebook --execute vae-iaf-cifar10.ipynb --ExecutePreprocessor.timeout=-1
