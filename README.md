---
jupyter:
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.10.9
  nbformat: 4
  nbformat_minor: 5
---

<div class="cell markdown">

# Breast Cancer Classification

## **Dataset Description:**

#### Breast cancer is the most common cancer amongst women in the world. It accounts for 25% of all cancer cases, and affected over 2.1 Million people in 2015 alone. It starts when cells in the breast begin to grow out of control. These cells usually form tumors that can be seen via X-ray or felt as lumps in the breast area.

#### The key challenges against it’s detection is how to classify tumors into malignant (cancerous) or benign(non cancerous). We ask you to complete the analysis of classifying these tumors using machine learning (with **SVMs**) and the Breast Cancer Wisconsin (Diagnostic) Dataset.

#### Dataset Link: **<https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset>**

<img src='https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQnjo50XZRwyzxm6PCF4tc7E-9S_wqdSKVm-Q&usqp=CAU' width=300>

</div>

<div class="cell markdown">

# Import Libraries

</div>

<div class="cell code" execution_count="60">

``` python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import plotly.express as px
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from summarytools import dfSummary
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
```

</div>

<div class="cell markdown">

# Read Breast Cancer Dataset

</div>

<div class="cell code" execution_count="2">

``` python
breast_cancer_df = pd.read_csv('breast-cancer.csv')
breast_cancer_df.drop(columns=['id'], inplace=True)
breast_cancer_df.sample(10)
```

<div class="output execute_result" execution_count="2">

        diagnosis  radius_mean  texture_mean  perimeter_mean  area_mean  \
    339         M       23.510         24.27          155.10     1747.0   
    175         B        8.671         14.45           54.42      227.2   
    322         B       12.860         13.32           82.82      504.8   
    123         B       14.500         10.89           94.28      640.7   
    386         B       12.210         14.09           78.78      462.0   
    474         B       10.880         15.62           70.41      358.9   
    440         B       10.970         17.20           71.73      371.5   
    93          B       13.450         18.30           86.60      555.1   
    96          B       12.180         17.84           77.79      451.1   
    518         B       12.880         18.22           84.45      493.1   

         smoothness_mean  compactness_mean  concavity_mean  concave points_mean  \
    339          0.10690           0.12830         0.23080              0.14100   
    175          0.09138           0.04276         0.00000              0.00000   
    322          0.11340           0.08834         0.03800              0.03400   
    123          0.11010           0.10990         0.08842              0.05778   
    386          0.08108           0.07823         0.06839              0.02534   
    474          0.10070           0.10690         0.05115              0.01571   
    440          0.08915           0.11130         0.09457              0.03613   
    93           0.10220           0.08165         0.03974              0.02780   
    96           0.10450           0.07057         0.02490              0.02941   
    518          0.12180           0.16610         0.04825              0.05303   

         symmetry_mean  ...  radius_worst  texture_worst  perimeter_worst  \
    339         0.1797  ...        30.670          30.73           202.40   
    175         0.1722  ...         9.262          17.04            58.36   
    322         0.1543  ...        14.040          21.08            92.80   
    123         0.1856  ...        15.700          15.98           102.80   
    386         0.1646  ...        13.130          19.29            87.65   
    474         0.1861  ...        11.940          19.35            80.78   
    440         0.1489  ...        12.360          26.87            90.14   
    93          0.1638  ...        15.100          25.94            97.59   
    96          0.1900  ...        12.830          20.92            82.14   
    518         0.1709  ...        15.050          24.37            99.31   

         area_worst  smoothness_worst  compactness_worst  concavity_worst  \
    339      2906.0            0.1515            0.26780           0.4819   
    175       259.2            0.1162            0.07057           0.0000   
    322       599.5            0.1547            0.22310           0.1791   
    123       745.5            0.1313            0.17880           0.2560   
    386       529.9            0.1026            0.24310           0.3076   
    474       433.1            0.1332            0.38980           0.3365   
    440       476.4            0.1391            0.40820           0.4779   
    93        699.4            0.1339            0.17510           0.1381   
    96        495.2            0.1140            0.09358           0.0498   
    518       674.7            0.1456            0.29610           0.1246   

         concave points_worst  symmetry_worst  fractal_dimension_worst  
    339               0.20890          0.2593                  0.07738  
    175               0.00000          0.2592                  0.07848  
    322               0.11550          0.2382                  0.08553  
    123               0.12210          0.2889                  0.08006  
    386               0.09140          0.2677                  0.08824  
    474               0.07966          0.2581                  0.10800  
    440               0.15550          0.2540                  0.09532  
    93                0.07911          0.2678                  0.06603  
    96                0.05882          0.2227                  0.07376  
    518               0.10960          0.2582                  0.08893  

    [10 rows x 31 columns]

</div>

</div>

<div class="cell markdown">

# Feature Engineering and EDA

</div>

<div class="cell code" execution_count="3">

``` python
X = breast_cancer_df.drop(columns=['diagnosis'])
y = breast_cancer_df['diagnosis']
```

</div>

<div class="cell markdown">

## Show Description about the Dataset

</div>

<div class="cell code" execution_count="4">

``` python
breast_cancer_df.info()
```

<div class="output stream stdout">

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 569 entries, 0 to 568
    Data columns (total 31 columns):
     #   Column                   Non-Null Count  Dtype  
    ---  ------                   --------------  -----  
     0   diagnosis                569 non-null    object 
     1   radius_mean              569 non-null    float64
     2   texture_mean             569 non-null    float64
     3   perimeter_mean           569 non-null    float64
     4   area_mean                569 non-null    float64
     5   smoothness_mean          569 non-null    float64
     6   compactness_mean         569 non-null    float64
     7   concavity_mean           569 non-null    float64
     8   concave points_mean      569 non-null    float64
     9   symmetry_mean            569 non-null    float64
     10  fractal_dimension_mean   569 non-null    float64
     11  radius_se                569 non-null    float64
     12  texture_se               569 non-null    float64
     13  perimeter_se             569 non-null    float64
     14  area_se                  569 non-null    float64
     15  smoothness_se            569 non-null    float64
     16  compactness_se           569 non-null    float64
     17  concavity_se             569 non-null    float64
     18  concave points_se        569 non-null    float64
     19  symmetry_se              569 non-null    float64
     20  fractal_dimension_se     569 non-null    float64
     21  radius_worst             569 non-null    float64
     22  texture_worst            569 non-null    float64
     23  perimeter_worst          569 non-null    float64
     24  area_worst               569 non-null    float64
     25  smoothness_worst         569 non-null    float64
     26  compactness_worst        569 non-null    float64
     27  concavity_worst          569 non-null    float64
     28  concave points_worst     569 non-null    float64
     29  symmetry_worst           569 non-null    float64
     30  fractal_dimension_worst  569 non-null    float64
    dtypes: float64(30), object(1)
    memory usage: 137.9+ KB

</div>

</div>

<div class="cell code" execution_count="5">

``` python
dfSummary(breast_cancer_df)
```

<div class="output execute_result" execution_count="5">

    <pandas.io.formats.style.Styler at 0x132ea882ef0>

</div>

</div>

<div class="cell code" execution_count="6">

``` python
plt.figure(figsize=(15, 5))
sns.histplot(X.radius_mean, bins=50, kde=True)
```

<div class="output execute_result" execution_count="6">

    <Axes: xlabel='radius_mean', ylabel='Count'>

</div>

<div class="output display_data">

![](9f1e80be4fa3071bcbe876fd7a679c744e40228d.png)

</div>

</div>

<div class="cell code" execution_count="7">

``` python
plt.figure(figsize=(15, 5))
sns.histplot(X.texture_mean, bins=50, kde=True)
```

<div class="output execute_result" execution_count="7">

    <Axes: xlabel='texture_mean', ylabel='Count'>

</div>

<div class="output display_data">

![](f6f562e6522a3fe6aed30ec8ac754f9771f4f7b8.png)

</div>

</div>

<div class="cell code" execution_count="8">

``` python
plt.figure(figsize=(15, 5))
sns.histplot(X.perimeter_mean, bins=50, kde=True)
```

<div class="output execute_result" execution_count="8">

    <Axes: xlabel='perimeter_mean', ylabel='Count'>

</div>

<div class="output display_data">

![](870d4011880ce82def1ad7c463c6f1893d1ac24a.png)

</div>

</div>

<div class="cell code" execution_count="9">

``` python
plt.figure(figsize=(15, 5))
sns.histplot(X.area_mean, bins=50, kde=True)
```

<div class="output execute_result" execution_count="9">

    <Axes: xlabel='area_mean', ylabel='Count'>

</div>

<div class="output display_data">

![](fa603cdba7e423bbd84aa452cce673fe84b6ed30.png)

</div>

</div>

<div class="cell code" execution_count="10">

``` python
plt.figure(figsize=(15, 5))
sns.histplot(X.smoothness_mean, bins=50, kde=True)
```

<div class="output execute_result" execution_count="10">

    <Axes: xlabel='smoothness_mean', ylabel='Count'>

</div>

<div class="output display_data">

![](1613338c03791f351e5b0f1c64ca2c79654d7546.png)

</div>

</div>

<div class="cell code" execution_count="11">

``` python
plt.figure(figsize=(15, 5))
sns.histplot(X.compactness_mean, bins=50, kde=True)
```

<div class="output execute_result" execution_count="11">

    <Axes: xlabel='compactness_mean', ylabel='Count'>

</div>

<div class="output display_data">

![](fcc6e355362f54fb09c61e23edef5e9ded15a275.png)

</div>

</div>

<div class="cell code" execution_count="12">

``` python
plt.figure(figsize=(15, 5))
sns.histplot(X.concavity_mean, bins=50, kde=True)
```

<div class="output execute_result" execution_count="12">

    <Axes: xlabel='concavity_mean', ylabel='Count'>

</div>

<div class="output display_data">

![](92e441921bfa5a7353610541ff476d56ec4db4ae.png)

</div>

</div>

<div class="cell code" execution_count="13">

``` python
plt.figure(figsize=(15, 5))
sns.histplot(X['concave points_mean'], bins=50, kde=True)
```

<div class="output execute_result" execution_count="13">

    <Axes: xlabel='concave points_mean', ylabel='Count'>

</div>

<div class="output display_data">

![](14cf41654f214cb7ac3d4e246434210489d581e1.png)

</div>

</div>

<div class="cell code" execution_count="14">

``` python
plt.figure(figsize=(15, 5))
sns.histplot(X.symmetry_mean, bins=50, kde=True)
```

<div class="output execute_result" execution_count="14">

    <Axes: xlabel='symmetry_mean', ylabel='Count'>

</div>

<div class="output display_data">

![](1f1d3be43772100ceae2e1f9af96d43cea5a9ed4.png)

</div>

</div>

<div class="cell code" execution_count="15">

``` python
plt.figure(figsize=(15, 5))
sns.histplot(X.fractal_dimension_mean, bins=50, kde=True)
```

<div class="output execute_result" execution_count="15">

    <Axes: xlabel='fractal_dimension_mean', ylabel='Count'>

</div>

<div class="output display_data">

![](edb46dc00fbcbd14aa6cb48721cdee3b681a862a.png)

</div>

</div>

<div class="cell code" execution_count="16">

``` python
plt.figure(figsize=(15, 5))
sns.histplot(X.radius_se, bins=50, kde=True)
```

<div class="output execute_result" execution_count="16">

    <Axes: xlabel='radius_se', ylabel='Count'>

</div>

<div class="output display_data">

![](17dfb0183dc1b3e6d30b9fb56e9a6d1dbb9762c2.png)

</div>

</div>

<div class="cell code" execution_count="17">

``` python
plt.figure(figsize=(15, 5))
sns.histplot(X.texture_se, bins=50, kde=True)
```

<div class="output execute_result" execution_count="17">

    <Axes: xlabel='texture_se', ylabel='Count'>

</div>

<div class="output display_data">

![](5cd15f757fdfb42ab839c5938d43cf05ad75afe8.png)

</div>

</div>

<div class="cell code" execution_count="18">

``` python
plt.figure(figsize=(15, 5))
sns.histplot(X.perimeter_se, bins=50, kde=True)
```

<div class="output execute_result" execution_count="18">

    <Axes: xlabel='perimeter_se', ylabel='Count'>

</div>

<div class="output display_data">

![](a7565ba72bb5730b27bde894a679bbe668c603ad.png)

</div>

</div>

<div class="cell code" execution_count="19">

``` python
plt.figure(figsize=(15, 5))
sns.histplot(X.area_se, bins=50, kde=True)
```

<div class="output execute_result" execution_count="19">

    <Axes: xlabel='area_se', ylabel='Count'>

</div>

<div class="output display_data">

![](14d13fc81e8f9320a15d5fada6a4023402c9df95.png)

</div>

</div>

<div class="cell code" execution_count="20">

``` python
plt.figure(figsize=(15, 5))
sns.histplot(X.smoothness_se, bins=50, kde=True)
```

<div class="output execute_result" execution_count="20">

    <Axes: xlabel='smoothness_se', ylabel='Count'>

</div>

<div class="output display_data">

![](99a07e40a7e7bd0a896e8c9001ad92671e057c42.png)

</div>

</div>

<div class="cell code" execution_count="21">

``` python
plt.figure(figsize=(15, 5))
sns.histplot(X.compactness_se, bins=50, kde=True)
```

<div class="output execute_result" execution_count="21">

    <Axes: xlabel='compactness_se', ylabel='Count'>

</div>

<div class="output display_data">

![](1dda39c8385007a3a0cc69bacd0802452e01af9e.png)

</div>

</div>

<div class="cell code" execution_count="22">

``` python
plt.figure(figsize=(15, 5))
sns.histplot(X.concavity_se, bins=50, kde=True)
```

<div class="output execute_result" execution_count="22">

    <Axes: xlabel='concavity_se', ylabel='Count'>

</div>

<div class="output display_data">

![](7510221d6ba84d50e4ac739475b26c9ca1672844.png)

</div>

</div>

<div class="cell code" execution_count="23">

``` python
plt.figure(figsize=(15, 5))
sns.histplot(X['concave points_se'], bins=50, kde=True)
```

<div class="output execute_result" execution_count="23">

    <Axes: xlabel='concave points_se', ylabel='Count'>

</div>

<div class="output display_data">

![](9e78f3a6bd761355cf37298ceec21b0ce7e1199d.png)

</div>

</div>

<div class="cell code" execution_count="24">

``` python
plt.figure(figsize=(15, 5))
sns.histplot(X.symmetry_se, bins=50, kde=True)
```

<div class="output execute_result" execution_count="24">

    <Axes: xlabel='symmetry_se', ylabel='Count'>

</div>

<div class="output display_data">

![](1a5117e8df2eaf2ddd5e6f71c44baf7ed46d16ce.png)

</div>

</div>

<div class="cell code" execution_count="25">

``` python
plt.figure(figsize=(15, 5))
sns.histplot(X.fractal_dimension_se, bins=50, kde=True)
```

<div class="output execute_result" execution_count="25">

    <Axes: xlabel='fractal_dimension_se', ylabel='Count'>

</div>

<div class="output display_data">

![](b28876b214c85fdbe20b2f75c52f828a99e2d2b2.png)

</div>

</div>

<div class="cell code" execution_count="26">

``` python
plt.figure(figsize=(15, 5))
sns.histplot(X.radius_worst, bins=50, kde=True)
```

<div class="output execute_result" execution_count="26">

    <Axes: xlabel='radius_worst', ylabel='Count'>

</div>

<div class="output display_data">

![](0a15325b3bce877914e9ab22ba4497c6bf21c4b2.png)

</div>

</div>

<div class="cell code" execution_count="27">

``` python
plt.figure(figsize=(15, 5))
sns.histplot(X.texture_worst, bins=50, kde=True)
```

<div class="output execute_result" execution_count="27">

    <Axes: xlabel='texture_worst', ylabel='Count'>

</div>

<div class="output display_data">

![](feb0c7dc17b92f34f23a1fde569915232c739be1.png)

</div>

</div>

<div class="cell code" execution_count="28">

``` python
plt.figure(figsize=(15, 5))
sns.histplot(X.perimeter_worst, bins=50, kde=True)
```

<div class="output execute_result" execution_count="28">

    <Axes: xlabel='perimeter_worst', ylabel='Count'>

</div>

<div class="output display_data">

![](161006b22bedd57d92ef9e0af662856d14804d42.png)

</div>

</div>

<div class="cell code" execution_count="29">

``` python
plt.figure(figsize=(15, 5))
sns.histplot(X.area_worst, bins=50, kde=True)
```

<div class="output execute_result" execution_count="29">

    <Axes: xlabel='area_worst', ylabel='Count'>

</div>

<div class="output display_data">

![](115405a6f866b13c9b66907429cfa9ff1eb03424.png)

</div>

</div>

<div class="cell code" execution_count="30">

``` python
plt.figure(figsize=(15, 5))
sns.histplot(X.smoothness_worst, bins=50, kde=True)
```

<div class="output execute_result" execution_count="30">

    <Axes: xlabel='smoothness_worst', ylabel='Count'>

</div>

<div class="output display_data">

![](5a7260866c7567934cc981cd60428b66bf51fbf8.png)

</div>

</div>

<div class="cell code" execution_count="31">

``` python
plt.figure(figsize=(15, 5))
sns.histplot(X.compactness_worst, bins=50, kde=True)
```

<div class="output execute_result" execution_count="31">

    <Axes: xlabel='compactness_worst', ylabel='Count'>

</div>

<div class="output display_data">

![](c9353ca19b9a92528a12d86e6ae971adc6a9e278.png)

</div>

</div>

<div class="cell code" execution_count="32">

``` python
plt.figure(figsize=(15, 5))
sns.histplot(X.concavity_worst, bins=50, kde=True)
```

<div class="output execute_result" execution_count="32">

    <Axes: xlabel='concavity_worst', ylabel='Count'>

</div>

<div class="output display_data">

![](baa26c5d6085a04d8d125ed9476d0ea8e4fd9996.png)

</div>

</div>

<div class="cell code" execution_count="33">

``` python
plt.figure(figsize=(15, 5))
sns.histplot(X['concave points_worst'], bins=50, kde=True)
```

<div class="output execute_result" execution_count="33">

    <Axes: xlabel='concave points_worst', ylabel='Count'>

</div>

<div class="output display_data">

![](68d3a480233b12d000633e60746483b6752f7de5.png)

</div>

</div>

<div class="cell code" execution_count="34">

``` python
plt.figure(figsize=(15, 5))
sns.histplot(X.symmetry_worst, bins=50, kde=True)
```

<div class="output execute_result" execution_count="34">

    <Axes: xlabel='symmetry_worst', ylabel='Count'>

</div>

<div class="output display_data">

![](e18b9bfb1cb51c0540dca40708614eff95abcc35.png)

</div>

</div>

<div class="cell code" execution_count="35">

``` python
plt.figure(figsize=(15, 5))
sns.histplot(X.fractal_dimension_worst, bins=50, kde=True)
```

<div class="output execute_result" execution_count="35">

    <Axes: xlabel='fractal_dimension_worst', ylabel='Count'>

</div>

<div class="output display_data">

![](ef56db625c3bcb5382cb5bb67468dadf0100f786.png)

</div>

</div>

<div class="cell code" execution_count="36">

``` python
plt.figure(figsize=(15, 5))
sns.boxenplot(X)
```

<div class="output execute_result" execution_count="36">

    <Axes: >

</div>

<div class="output display_data">

![](5eee0ffa661df1a39bf5ab8e2ca05a538b821caf.png)

</div>

</div>

<div class="cell markdown">

## Plot The Destripution of The Dataset

</div>

<div class="cell code" execution_count="56">

``` python
pca = PCA(n_components=2, random_state=42)
decomposition_features = pca.fit_transform(X)
```

</div>

<div class="cell code" execution_count="57">

``` python
feature_1 = decomposition_features[:, 0]
feature_2 = decomposition_features[:, 1]
```

</div>

<div class="cell code" execution_count="64">

``` python
px.scatter(
    x=feature_1,
    y=feature_2,
    color=breast_cancer_df['diagnosis'],
    title='Breast Cancer Dataset in 2D'
)
```

<div class="output display_data">

![](4f78f7a0f58933393875431bf11c64b085d93754.png)

</div>

</div>

<div class="cell code" execution_count="65">

``` python
plt.figure(figsize=(15, 8))
sns.jointplot(
    x=feature_1,
    y=feature_2,
    hue=breast_cancer_df['diagnosis']
)
```

<div class="output execute_result" execution_count="65">

    <seaborn.axisgrid.JointGrid at 0x132fd17efe0>

</div>

<div class="output display_data">

    <Figure size 1500x800 with 0 Axes>

</div>

<div class="output display_data">

![](356b81e54f4d00d68e504a9c62978a6be205c41b.png)

</div>

</div>

<div class="cell markdown">

## Check Data Balanced Or Not?

</div>

<div class="cell code" execution_count="41">

``` python
y.value_counts(normalize=True).plot.pie(explode=[0.1, 0], autopct='%1.2f%%')
```

<div class="output execute_result" execution_count="41">

    <Axes: ylabel='diagnosis'>

</div>

<div class="output display_data">

![](8ede59939847b3a056199b9cb05734f396bbe7d1.png)

</div>

</div>

<div class="cell markdown">

## Scale data using Robust Scaler

</div>

<div class="cell code" execution_count="42">

``` python
robust_scaler = RobustScaler()
X = robust_scaler.fit_transform(X)
```

</div>

<div class="cell code" execution_count="43">

``` python
X
```

<div class="output execute_result" execution_count="43">

    array([[ 1.13235294, -1.5026643 ,  1.26374006, ...,  1.71524826,
             2.63555556,  1.88457808],
           [ 1.76470588, -0.19005329,  1.61285862, ...,  0.89219446,
            -0.10666667,  0.43549952],
           [ 1.54901961,  0.42806394,  1.51261666, ...,  1.48305173,
             1.17185185,  0.3656644 ],
           ...,
           [ 0.79166667,  1.64120782,  0.76253025, ...,  0.43402094,
            -0.89481481, -0.08923375],
           [ 1.77205882,  1.86323268,  1.86173522, ...,  1.7111019 ,
             1.87407407,  2.13191077],
           [-1.375     ,  1.01243339, -1.32457656, ..., -1.03586607,
             0.07259259, -0.46799224]])

</div>

</div>

<div class="cell markdown">

## Split dataset into 80% Training and % Testing

</div>

<div class="cell markdown">

### Use Label Encodwer to encode our Target

</div>

<div class="cell code" execution_count="44">

``` python
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
```

</div>

<div class="cell markdown">

### Split Data

</div>

<div class="cell code" execution_count="45">

``` python
x_train, x_test, y_train, y_test = train_test_split(
    X,
    y,
    random_state=0,
    shuffle=True,
    test_size=0.2,
    stratify=y,
)
```

</div>

<div class="cell markdown">

# SVM Model

</div>

<div class="cell code" execution_count="46">

``` python
svc = SVC(random_state=0)
svc.fit(x_train, y_train)
```

<div class="output execute_result" execution_count="46">

    SVC(random_state=0)

</div>

</div>

<div class="cell code" execution_count="47">

``` python
svc_y_predict = svc.predict(x_test)
```

</div>

<div class="cell code" execution_count="48">

``` python
print(metrics.classification_report(y_test, svc_y_predict)) 
```

<div class="output stream stdout">

                  precision    recall  f1-score   support

               0       0.97      0.97      0.97        72
               1       0.95      0.95      0.95        42

        accuracy                           0.96       114
       macro avg       0.96      0.96      0.96       114
    weighted avg       0.96      0.96      0.96       114

</div>

</div>

<div class="cell code" execution_count="49">

``` python
metrics.ConfusionMatrixDisplay.from_estimator(svc, x_test, y_test)
```

<div class="output execute_result" execution_count="49">

    <sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x132f29ea2c0>

</div>

<div class="output display_data">

![](98cffdf5845497479ca09cf554cc1323c8ffeb2e.png)

</div>

</div>

<div class="cell code" execution_count="50">

``` python
round(
    metrics.accuracy_score(y_true=y_test, y_pred=svc_y_predict) * 100,
    4
)
```

<div class="output execute_result" execution_count="50">

    96.4912

</div>

</div>

<div class="cell code" execution_count="51">

``` python
metrics.RocCurveDisplay.from_estimator(
    svc,
    x_test,
    y_test
)
```

<div class="output execute_result" execution_count="51">

    <sklearn.metrics._plot.roc_curve.RocCurveDisplay at 0x132f2e21a20>

</div>

<div class="output display_data">

![](d023cd9f16f15cb6ae778e323ba4a4827bc8584e.png)

</div>

</div>

<div class="cell code" execution_count="52">

``` python
metrics.PrecisionRecallDisplay.from_estimator(
    svc,
    x_test,
    y_test
)
```

<div class="output execute_result" execution_count="52">

    <sklearn.metrics._plot.precision_recall_curve.PrecisionRecallDisplay at 0x132f6b891b0>

</div>

<div class="output display_data">

![](35e3371e89cc5ba50108a8a22a4ea33d97b2de5a.png)

</div>

</div>

<div class="cell code" execution_count="53">

``` python
metrics.precision_score(y_test, svc_y_predict).round(3)
```

<div class="output execute_result" execution_count="53">

    0.952

</div>

</div>

<div class="cell code" execution_count="54">

``` python
metrics.recall_score(y_test, svc_y_predict).round(3)
```

<div class="output execute_result" execution_count="54">

    0.952

</div>

</div>

<div class="cell code" execution_count="55">

``` python
metrics.f1_score(y_test, svc_y_predict).round(2)
```

<div class="output execute_result" execution_count="55">

    0.95

</div>

</div>

<div class="cell markdown">

## Tune SVC Model

</div>

<div class="cell code" execution_count="61">

``` python
parameters = {'kernel' : ('linear', 'rbf'), 'C' : range(1, 11)}
```

</div>

<div class="cell code" execution_count="62">

``` python
clf = GridSearchCV(svc, parameters)
```

</div>

<div class="cell code" execution_count="63">

``` python
clf.fit(x_train, y_train)
```

<div class="output execute_result" execution_count="63">

    GridSearchCV(estimator=SVC(random_state=0),
                 param_grid={'C': range(1, 11), 'kernel': ('linear', 'rbf')})

</div>

</div>

<div class="cell code" execution_count="66">

``` python
clf.best_params_
```

<div class="output execute_result" execution_count="66">

    {'C': 2, 'kernel': 'rbf'}

</div>

</div>

<div class="cell code" execution_count="73">

``` python
clf.cv_results_
```

<div class="output execute_result" execution_count="73">

    {'mean_fit_time': array([0.00199876, 0.00279984, 0.00220189, 0.00260072, 0.00270233,
            0.00199804, 0.00240088, 0.00259943, 0.00239902, 0.00240197,
            0.00230212, 0.00180001, 0.0022018 , 0.00199857, 0.00259705,
            0.00220032, 0.00229998, 0.00219846, 0.00259604, 0.00199785]),
     'std_fit_time': array([3.25369303e-06, 3.99882606e-04, 7.47923219e-04, 4.89341606e-04,
            4.00591961e-04, 1.26698782e-06, 4.88715565e-04, 4.88873850e-04,
            4.90330943e-04, 4.88794595e-04, 4.02890729e-04, 4.01617081e-04,
            7.42855930e-04, 6.28207568e-04, 4.90154259e-04, 4.02361495e-04,
            5.98838056e-04, 3.99472645e-04, 1.01681300e-03, 3.84320100e-06]),
     'mean_score_time': array([0.00080199, 0.00099945, 0.00040011, 0.00119963, 0.00080228,
            0.00120053, 0.00079975, 0.00120034, 0.00080075, 0.00119934,
            0.00060043, 0.00099735, 0.00059805, 0.00100193, 0.00040164,
            0.00099993, 0.00080223, 0.0008029 , 0.00080075, 0.00059943]),
     'std_score_time': array([4.01007928e-04, 1.45415789e-06, 4.90038112e-04, 3.99414630e-04,
            4.01157209e-04, 4.00376643e-04, 3.99876067e-04, 3.99042810e-04,
            4.00376927e-04, 4.00139361e-04, 4.90252017e-04, 3.25997632e-06,
            4.88326454e-04, 6.37030326e-04, 4.91914364e-04, 3.01578299e-07,
            4.01129504e-04, 4.01483596e-04, 4.00383202e-04, 4.89434505e-04]),
     'param_C': masked_array(data=[1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9,
                        10, 10],
                  mask=[False, False, False, False, False, False, False, False,
                        False, False, False, False, False, False, False, False,
                        False, False, False, False],
            fill_value='?',
                 dtype=object),
     'param_kernel': masked_array(data=['linear', 'rbf', 'linear', 'rbf', 'linear', 'rbf',
                        'linear', 'rbf', 'linear', 'rbf', 'linear', 'rbf',
                        'linear', 'rbf', 'linear', 'rbf', 'linear', 'rbf',
                        'linear', 'rbf'],
                  mask=[False, False, False, False, False, False, False, False,
                        False, False, False, False, False, False, False, False,
                        False, False, False, False],
            fill_value='?',
                 dtype=object),
     'params': [{'C': 1, 'kernel': 'linear'},
      {'C': 1, 'kernel': 'rbf'},
      {'C': 2, 'kernel': 'linear'},
      {'C': 2, 'kernel': 'rbf'},
      {'C': 3, 'kernel': 'linear'},
      {'C': 3, 'kernel': 'rbf'},
      {'C': 4, 'kernel': 'linear'},
      {'C': 4, 'kernel': 'rbf'},
      {'C': 5, 'kernel': 'linear'},
      {'C': 5, 'kernel': 'rbf'},
      {'C': 6, 'kernel': 'linear'},
      {'C': 6, 'kernel': 'rbf'},
      {'C': 7, 'kernel': 'linear'},
      {'C': 7, 'kernel': 'rbf'},
      {'C': 8, 'kernel': 'linear'},
      {'C': 8, 'kernel': 'rbf'},
      {'C': 9, 'kernel': 'linear'},
      {'C': 9, 'kernel': 'rbf'},
      {'C': 10, 'kernel': 'linear'},
      {'C': 10, 'kernel': 'rbf'}],
     'split0_test_score': array([0.98901099, 0.97802198, 0.97802198, 0.97802198, 0.97802198,
            0.97802198, 0.97802198, 0.97802198, 0.96703297, 0.97802198,
            0.96703297, 0.97802198, 0.96703297, 0.97802198, 0.96703297,
            0.96703297, 0.96703297, 0.96703297, 0.96703297, 0.96703297]),
     'split1_test_score': array([0.95604396, 0.96703297, 0.95604396, 0.97802198, 0.95604396,
            0.96703297, 0.95604396, 0.96703297, 0.95604396, 0.96703297,
            0.96703297, 0.96703297, 0.98901099, 0.96703297, 0.98901099,
            0.96703297, 0.98901099, 0.96703297, 0.98901099, 0.96703297]),
     'split2_test_score': array([0.97802198, 0.98901099, 0.98901099, 0.97802198, 0.97802198,
            0.98901099, 0.97802198, 0.98901099, 0.96703297, 1.        ,
            0.96703297, 0.98901099, 0.96703297, 0.98901099, 0.96703297,
            0.98901099, 0.96703297, 0.97802198, 0.96703297, 0.97802198]),
     'split3_test_score': array([0.95604396, 0.95604396, 0.96703297, 0.96703297, 0.97802198,
            0.96703297, 0.96703297, 0.96703297, 0.97802198, 0.96703297,
            0.97802198, 0.96703297, 0.95604396, 0.96703297, 0.95604396,
            0.96703297, 0.94505495, 0.96703297, 0.94505495, 0.95604396]),
     'split4_test_score': array([0.95604396, 0.97802198, 0.95604396, 0.98901099, 0.94505495,
            0.98901099, 0.94505495, 0.98901099, 0.94505495, 0.97802198,
            0.94505495, 0.97802198, 0.94505495, 0.97802198, 0.94505495,
            0.97802198, 0.93406593, 0.97802198, 0.93406593, 0.97802198]),
     'mean_test_score': array([0.96703297, 0.97362637, 0.96923077, 0.97802198, 0.96703297,
            0.97802198, 0.96483516, 0.97802198, 0.96263736, 0.97802198,
            0.96483516, 0.97582418, 0.96483516, 0.97582418, 0.96483516,
            0.97362637, 0.96043956, 0.97142857, 0.96043956, 0.96923077]),
     'std_test_score': array([0.01390012, 0.01120664, 0.01281528, 0.00695006, 0.01390012,
            0.00982887, 0.01281528, 0.00982887, 0.01120664, 0.01203786,
            0.01076699, 0.00822342, 0.01457857, 0.00822342, 0.01457857,
            0.00879121, 0.01916   , 0.00538349, 0.01916   , 0.00822342]),
     'rank_test_score': array([12,  7, 10,  1, 13,  1, 16,  1, 18,  1, 16,  5, 14,  5, 14,  7, 19,
             9, 19, 11])}

</div>

</div>

<div class="cell code" execution_count="67">

``` python
best_svc = SVC(kernel='rbf', C=2)
best_svc.fit(x_train, y_train)
```

<div class="output execute_result" execution_count="67">

    SVC(C=2)

</div>

</div>

<div class="cell code" execution_count="68">

``` python
y_pred = best_svc.predict(x_test)
```

</div>

<div class="cell code" execution_count="69">

``` python
metrics.ConfusionMatrixDisplay.from_estimator(best_svc, x_test, y_test)
```

<div class="output execute_result" execution_count="69">

    <sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x132feabe410>

</div>

<div class="output display_data">

![](98cffdf5845497479ca09cf554cc1323c8ffeb2e.png)

</div>

</div>

<div class="cell code" execution_count="70">

``` python
print(metrics.classification_report(y_test, y_pred))
```

<div class="output stream stdout">

                  precision    recall  f1-score   support

               0       0.97      0.97      0.97        72
               1       0.95      0.95      0.95        42

        accuracy                           0.96       114
       macro avg       0.96      0.96      0.96       114
    weighted avg       0.96      0.96      0.96       114

</div>

</div>

<div class="cell code" execution_count="71">

``` python
metrics.accuracy_score(y_test, y_pred).round(3)
```

<div class="output execute_result" execution_count="71">

    0.965

</div>

</div>
