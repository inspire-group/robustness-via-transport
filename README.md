# Lower Bounds on Robustness from Optimal Transport

This code accompanies the paper 'Lower Bounds on Adversarial Robustness from Optimal Transport' which has been accepted to NeurIPS 2019. It assumes that the Fashion MNIST data has been downloaded to /home/data/ on the user's machine.

Dependencies: Tensorflow-1.8, keras, numpy, scipy, scikit-learn

To compute the bound for _n_ samples from a dataset D and with the norm P, run

```
python compute_cost.py --dataset=D --norm=P --no_of_examples=n
```
We support D=MNIST, fMNIST and CIFAR-10 as well as P=l2 or linf. 

To train a robust model on dataset D using Projected Gradient Descent with parameters of your choice, run
```
python train_adv.py model_dir/modelA_large_adv --dataset=D --eps=e --type=i --norm=P --delta=del --num_iter=ni --epochs=ep --iter=1 --two_class --noise
```
We support D=MNIST and fMNIST as well as P=l2 or linf. The `two-class` flag ensures a binary classification problem is being solved.

To run attack A on the model, with parameters of your choice but the same norm P, run
```
python test_adv.py A model_dir/modelA_large_adv_D_e_P_ep_iter_ni_del_nob_3_7_noise --dataset=D --eps=e2 --norm=P --delta=del2 --num_iter=ni2 --two_class --noise
```

Robust training on CIFAR-10 will be added soon.