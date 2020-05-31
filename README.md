# Example of PyTorch
2020-06-01  Kyo Kutsuzawa


## Summary
This is an example of PyTorch.
In this example, a feed-forward neural network is implemented to approximate a sin-wave function.


## Usage

### Dataset generation

Generate a training dataset:
```
$ python src/generate_dataset.py --n-sample 5000 --out dataset/train.csv
```

Generate a validation dataset:
```
$ python src/generate_dataset.py --n-sample 500 --out dataset/validation.csv
```

Generate a test dataset:
```
$ python src/generate_dataset.py --n-sample 500 --out dataset/test.csv
```


### Training

```
python src/train_custom_loop.py
```

There is another training code, `train.py`, that uses an external training tool named ignite.
It may be beneficial to some developers, but the custom-loop example is better for beginners.


### Evaluation

```
python3 src/evaluate.py
```


