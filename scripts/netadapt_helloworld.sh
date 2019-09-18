python master.py models/helloworld/netadapt 3 32 32 \
    -gp 0 1 2 -mi 3 -bur 0.25 -rt FLOPS \
    -irr 0.025 -rd 1.0 -lr 0.001 -st 5 \
    -im models/helloworld/model_0.pth.tar \
    -lt models/helloworld/lut.pkl  -dp data/ \
    --arch helloworld -si 1