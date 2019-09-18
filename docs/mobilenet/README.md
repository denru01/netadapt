1. **Training MobileNet on CIFAR-10.**

    Training:
    ```bash
    python train.py data/ --dir models/mobilenet/model.pth.tar --arch mobilenet
    ```
    
    Evaluation:
    ```bash
    python eval.py data/ --dir models/mobilenet/model.pth.tar --arch mobilenet
    ```
    
    One trained model can be found [here](https://drive.google.com/file/d/1jtRZOHK1daRTKD4jYu4US84lf8YjqEdJ/view?usp=sharing).
     
    
2. **Measuring Latency**

    Here we build the latency lookup table for `cuda:0` device:
    ```bash
    python build_lookup_table.py --dir latency_lut/lut_mobilenet.pkl --arch mobilenet
    ```
    It measures latency of different layers contained in the network (i.e. **MobileNet** here).
    For conv layers, the sampled numbers of feature channels are multiples of `MIN_CONV_FEATURE_SIZE`.
    For fc layers, the sampled numbers of features are multiples of `MIN_FC_FEATURE_SIZE`. 
        
3. **Applying NetAdapt**

    Modify which GPUs will be utilized (`-gp`) in `netadapt_mobilenet-0.5mac.sh` and run the script to apply NetAdapt to a pretrained model:
    ```bash
    sh scripts/netadapt_mobilenet-0.5mac.sh
    ```
    
    You can see how the model is simplified at each iteration in `models/mobilenet/prune-by-mac/master/history.txt` and
    select the one that satisfies the constraints to run long-term fine-tune.
    
    After obtaining the adapted model, we need to finetune the model (here we select the one after 28 iterations):
    ```bash
    python train.py data/ --arch mobilenet --resume models/mobilenet/prune-by-mac/master/iter_28_best_model.pth.tar --dir models/mobilenet/prune-by-mac/master/finetune_model.pth.tar --lr 0.001
    ```
    
    <p align="center">
	<img src="../../fig/netadapt_algo.png" alt="photo not available" width="90%" height="90%">
    </p>
    
    
    If you want to get a model with 50% latency, please run:
    ```bash
    sh scripts/netadapt_mobilenet-0.5latency.sh
    ```
    

4. **Evaluation Using Adapted Models**

    After applying NetAdapt to a pretrained model, we can evaluate this adapted model using:
    ```bash
    python eval.py data/ --dir models/mobilenet/prune-by-mac/master/finetune_model.pth.tar --arch mobilenet
    ```
    
    The adapted model can be restored **without modifying the orignal python file**.
    
    We provide one adapted model [here](https://drive.google.com/file/d/1wkQPolgv34ESb0gyeYdygbuNuNYhIqyx/view?usp=sharing).