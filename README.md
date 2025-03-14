# Helper Library for Machine Learning and Data Manipulation

A small collection of functions designed for data processing, numerical transformations, and visualization in machine learning workflows.

## Functions Overview + Usage

### 1. number_to_binary(number, num_bits=16)
Converts an integer to its binary representation as a NumPy array.

```python
>>> number_to_binary(5)
array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1])

>>> number_to_binary(5, num_bits=8)
array([0, 0, 0, 0, 0, 1, 0, 1])
```

### 2. binary_to_number(binary_array)
Converts a NumPy array representing a binary number to its decimal equivalent.

```python
>>> binary_to_number(np.array([1, 0, 1, 1]))
11
```

### 3. calc_accuracy(outputs, labels)
Calculates the accuracy of predictions compared to the true labels.

```python
>>> outputs = np.array([0, 1, 1, 0, 1])
>>> labels = np.array([0, 1, 0, 0, 1])
>>> calc_accuracy(outputs, labels)
80.00
```

### 4. normalize_array(arr)
Normalizes a NumPy array to the range 0-1.

```python
>>> arr = np.array([10, 20, 30, 40, 50])
>>> normalize_array(arr)
array([0., 0.25, 0.5, 0.75, 1.])
```

### 5. print_train_info(current_epoch, total_epochs, loss)
Prints the current epoch, total epochs, and loss in a formatted and readable way.

```python
>>> print_train_info(2, 10, 0.3456)
Epoch: [2 / 10], Loss: 0.3456
```

### 6. split_array(array, train_ratio=0.8)
Splits an array into training and validation subsets.

```python
>>> array = np.array([1, 2, 3, 4, 5])
>>> train, val = split_array(array, train_ratio=0.6)
>>> train, val
(array([1, 2, 3]), array([4, 5]))
```

### 7. plot_losses(epoch_losses, every_x_epoch=1)
Plots loss values for every specified epoch interval.

```python
>>> losses = [0.8, 0.6, 0.5, 0.3, 0.2, 0.1]
>>> plot_losses(losses, 2)  # Plots losses for epochs 2, 4, 6
```

### 8. add_custom_features(array)
Expands an input array with custom features: 
𝑥**2
, 
𝑥**3
, sign(x), 
𝑒**𝑥
, cos(x), and a sign indicator.

```python
>>> arr = np.array([-2, -1, 0, 1, 2])
>>> add_custom_features(arr)
array([[-2., 4., -8., -1., 0.13533528, -0.41614684, -1.],
       [-1., 1., -1., -1., 0.36787944, 0.54030231, -1.],
       [ 0., 0., 0., 0., 1., 1., 0.],
       [ 1., 1., 1., 1., 2.71828183, 0.54030231, 1.],
       [ 2., 4., 8., 1., 7.3890561 , -0.41614684, 1.]])
```

### 9. standardize_array(array)
Standardizes an array to have a mean of 0 and standard deviation of 1.

```python
>>> arr = np.array([10, 20, 30, 40, 50])
>>> standardized_arr = standardize_array(arr)
>>> standardized_arr
array([-1.41421356, -0.70710678, 0., 0.70710678, 1.41421356])
```

### 10. min_max_scale(array, a=0, b=1)
Scales an array to the range [a, b].

```python
>>> arr = np.array([10, 20, 30, 40, 50])
>>> scaled_arr = min_max_scale(arr, a=0, b=1)
>>> scaled_arr
array([0., 0.25, 0.5, 0.75, 1.])
```

---
## Author
- **Esmail Jamshidiasl (NeuralNinjaEsmail)**

## Co-author
- **Microsoft Copilot (AI Assistant)**

## License
MIT License