# EmoInt-Emotion-Classification
Emotion Classification in EmoInt Dataset
<hr/>

## Emotion Classification
EmoInt Dataset is for task on Emotion Intensity (EmoInt), however I used for task on **Emotion Classification**.<br>
I do not use intensity, **only tweet for input and emotion for output**.

## Model
Conv1D - MaxPool1D - Dense(1000) - Dense(4)

```
batch size: 4
conv kernel: 5
epochs: 10
dropout rate: 0.5
loss: categorical cross-entropy
optimizer: adam
```

## Result
|       | Loss   | Accuracy |
| :---: | :----: | :------: |
| Train | 0.1228 | 0.9678   |
| Test  | 1.0312 | 0.8132   |

To Do: Solve overfitting problem
