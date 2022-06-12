# Sentiment analysis using UCI ML Drug Reviews 

When performing any big data analysis, there are five major questions one needs to answer in order to have a complete data flow.
1. [What is the dataset](#what-is-the-dataset)?
2. [What are the preprocessing methods](#what-are-the-preprocessing-methods)?
3. [What are the models](#what-are-the-models)?
4. [What is the accuracy and precision](#what-is-the-accuracy-and-precision)?
5. [What is the output of the dataset](#what-is-the-output-of-the-dataset)? (optional) 

### What is the dataset?

In this article, I have used [UCI ML Drug Reviews](https://www.kaggle.com/datasets/jessicali9530/kuc-hackathon-winter-2018) from Kaggle.com. Kaggle.com is a data analysis website that contains a plethora of datasets and competitions online. You might find useful implementations on how to do preprocessing and modelling, saving some effort in answering questions 2 and 3.

A typical dataset usually is mixed with numerials, date and strings. Your job is to convert all the date and strings into numerials, as preprocessing libraries and models only understand numbers.

Remember this golden rule.

> **All csv data needs to be in numerical format.**


Some dataset does not contain files like `test.csv`, `train.csv` or `validation.csv`. If they are bundled as a single file, you should find some methods to split the csv file. For details, you can check out [pandas](https://pandas.pydata.org/). Typically, programmers will use `train_test_split()` to split the dataset. [Example](https://www.geeksforgeeks.org/how-to-split-a-dataset-into-train-and-test-sets-using-python/).

[click on this link](#my-multi-word-header)


### What are the preprocessing methods?

As mentioned on the previous section, all data must be in numerical value. Some columns may not be in numerial number in the first place. To do that, there are several ready to use libraries to convert different types of data into numerials. Here is an example list (non-exhaustive)

1. NLTK `from nltk.corpus import stopwords`
2. TextBlob
3. sklearn `from sklearn.preprocessing import LabelEncoder`

To generate even more data columns, you can also try (if applicable):
1. _regular expressions_ to remove special characters, non-ASCII characters. 
2. _lambda functions_ to count the number of words.

, before parsing into the preprocessing libraries. This can increase the number of columns (or dimensions in data science), so that it can have more training parameters. Usually, more training parameters the better.

### What are the models?
You can find a plethora of different machine learning models. From the most basic decision trees to the most advanced or state-of-the-art models. I use Random Forest and AdaBoost for training.

To input the cleaned data into these models. Simply assign `model.fit(X_train, Y_train)` to it and then `predict(X_val)`

For instance, if you want to use Random Forest to predict models.
```python
#fit the model and predicct the output
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix #import confusion_matrix

clf = RandomForestClassifier().fit(X_train, Y_train)

pred = clf.predict(X_val)

print("Accuracy: %s" % str(clf.score(X_val, Y_val)))
print("Confusion Matrix")
print(confusion_matrix(pred, Y_val))
```
```
Accuracy: 0.5462885738115096 
Confusion Matrix 
[[101 30 24 21 44]
[ 2 4 0 0 1]
[ 2 0 1 0 4]
[ 3 2 2 8 5]
[118 47 76 163 541]]
```


### What is the accuracy and precision?

This is quite straightforward, the higher the accuracy the better the model perform. I use mirco f1 and macro f1 to evaluate the score.
The are also other f1 metrics to evalute the model performance. The [sklearn.metrics.f1_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html) is a good place to start.

```python
from sklearn.metrics import f1_score
print(f1_score(Y_val, pred, average='macro'))
print(f1_score(Y_val, pred, average='micro'))
```

```
0.2502066764312927 # macro
0.5529608006672226 # micro
```


### What is the output of the dataset?
If you want to get the output, simply use `pandas` to make it as a dataframe and export to csv.

```python
df['result'] = pd.DataFrame(data=pred)

df.to_csv(index=False, path_or_buf='result.csv')
```

























### My Multi Word Header