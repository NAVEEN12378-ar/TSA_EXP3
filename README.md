# Ex.No: 03   COMPUTE THE AUTO CORRELATION FUNCTION(ACF)
Date: 08-04-2025

### AIM:
To Compute the AutoCorrelation Function (ACF) of the data for the first 35 lags to determine the model
type to fit the data.
### ALGORITHM:
1. Import the necessary packages
2. Find the mean, variance and then implement normalization for the data.
3. Implement the correlation using necessary logic and obtain the results
4. Store the results in an array
5. Represent the result in graphical representation as given below.
### PROGRAM:

```
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

X = [3, 16, 156, 47, 246, 176, 233, 140, 130, 101, 166, 201, 200, 116, 118, 247, 209, 52, 153, 232, 128, 27, 192, 168, 208, 187, 228, 86, 30, 151, 18, 254, 76, 112, 67, 244, 179, 150, 89, 49, 83, 147, 90, 33, 6, 158, 80, 35, 186, 127]

x_mean = np.mean(X)

x_var = np.var(X)
Normalized_data = (X - x_mean) / np.sqrt(x_var)
Acf_result = np.correlate(Normalized_data, Normalized_data, mode='full')
Acf_result = Acf_result[len(Acf_result)//2:]

plt.figure(figsize=(11, 5))
plt.stem(Acf_result[:36], use_line_collection=True)
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation Function->(ACF)')
plt.show()
```
### OUTPUT:

![exp3](https://github.com/user-attachments/assets/35330392-4aaf-405f-8551-25283f0864c0)

### RESULT:
        Thus we have successfully implemented the auto correlation function in python.
