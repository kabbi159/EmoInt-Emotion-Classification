# EmoInt-Emotion-Classification
Emotion Classification in EmoInt Dataset
<hr/>

## Emotion Classification
EmoInt Dataset is for task on Emotion Intensity (EmoInt), however I used for task on **Emotion Classification**.<br>
I do not use intensity, **only tweet for input and emotion for output**.

## Model
Conv1D - MaxPool1D - Dense(1000) - Dense(4)

```
batch size: 8
conv kernel: 5
epochs: 8
dropout rate: 0.5
loss: categorical cross-entropy
optimizer: adam
```

## Result
|       | Loss   | Accuracy |
| :---: | :----: | :------: |
| Train | 0.0451 | 0.9846   |
| Test  | 0.8364 | 0.7952   |

To Do: Solve overfitting problem
