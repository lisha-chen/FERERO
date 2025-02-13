## Experiment on Multi-MNIST and Fashion dataset

First run the training scripts to obtain the baseline results from prior works:

```
python individual_train.py
python linscalar_train.py
python epo_train.py
python pmtl_train.py
```


Then run the script to obtain the result for FERERO:
```
python ferero_train.py
```

This will create `.pkl` files in the `results` folder. Then use `display_result.py` to obtain the figures.