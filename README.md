# SLIM and SSLIM Recommendation Methods - TOY implementations

We have implemented SLIM [1] and SSLIM [2] (more specifically cSLIM) in a "toy" way, that's useful **only** to study these methods, it is absolutely **impossible** to use these implementations on production, because of performance and some static aspects put on the code.

The code implemented here is also based on mrec (https://github.com/Mendeley/mrec). We preferred to implement a toy version because we find some difficulties to debug mrec code due to its complexities related to being a production tool. If you want to use it in production, we really suggest you to go to this excellent implementation of SLIM done by the Mendeley Team.


We've also implemented some new ideas to extend these methods in some specific cases.

# Installing
Just install the requirements (it might take some minutes):

```python
git clone https://github.com/ruhan/toyslim
pip install -r requirements.txt
```

And after that, you can run any one of the four available algorithms.


# Running

## SLIM
All train and test files would be stored in the TSV format (Tab-separated values).
The value will be in the following order: user <TAB> item <TAB> value.

We have two SLIM implementations:

### SLIM single core
It is a simple implementation of SLIM. It is good to understand how it works. You can run it by:

```bash
python slim.py --train train.tsv --test test.tsv --output output.json
```

### SLIM multicore (parallel)
It is a SLIM implementation using all cores available on the machine.
It uses the ability of SLIM of being intrinsicly parallel. You can run it by:

```bash
python slim_parallel.py --train train.tsv --test test.tsv --output output.json
```

### SLIM Oracle
We also have an oracle implementation that gives us **the best possible** result we would have if our method was an "oracle". It is very useful to use as an upper boundary of any re-ranking algorithm.


```bash
python slim_oracle.py --train train.tsv --test test.tsv --output output.json
```


## SSLIM
All train, test, and side information files would be stored in the TSV format (Tab-separated values).
The value will be in the following order: user <TAB> item <TAB> value.


### SSLIM single core
It is a simple implementation of SLIM. It is good to understand how it works. You can run it by:

```bash
python sslim.py --train train.tsv --test test.tsv --side_information side_information.tsv --output output.json
```

### SSLIM multicore (parallel)
It is a SLIM implementation using all cores available on the machine. It uses the ability of SLIM of being intrinsically parallel. You can run it by:

```bash
python sslim_parallel.py --train train.tsv --test test.tsv --side_information side_information.tsv --output output.json
```

### SSLIM oracle
We also have an oracle implementation that gives us **the best possible** result we would have if our method was an "oracle". It is very useful to use as an upper boundary of any re-ranking algorithm.

```bash
python sslim_oracle.py --train train.tsv --test test.tsv --side_information=side_information.tsv --output output.json
```

## Data Preparation
We have two **very** simple scripts to make your life easy if you need to split your data into "train/test" (80%-20%). You can use them by executing:


### Train/Test
```bash
python split_data_train_test.py YOUR_USER_ITEM_DATA.tsv
```

### Train/Validation/Test
If you want to split your data into train/validation/test (60%-20%-20%) you should use this other one:

```bash
python split_data_train_validation_test.py YOUR_USER_ITEM_DATA.tsv
```

Again, it is only a plus, because the scripts has no flexibility. If you need to change the percentages you can make it easily by changing only an aspect in the code. In the future these scripts will also accept parameters like this.


# References
[1] Xia Ning and George Karypis (2011). SLIM: Sparse Linear Methods for Top-N Recommender Systems. ICDM, 2011.

[2] Xia Ning and George Karypis (2012). Sparse linear methods with side information for top-n recommendations. RecSys 2012.
