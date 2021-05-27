# FeedGate 

# Intro
DKT는 `Deep Knowledge Tracing`의 약자로 우리의 "지식 상태"를 추적하는 딥러닝 방법론입니다. DKT를 활용한다면 학생들의 과목별 이해도를 분석할 수 있고, 취약한 부분을 극복하기 위해 어떤 문제들을 풀면 좋을지 추천이 가능합니다. 따라서 DKT는 교육 AI의 추천이라고 불리울 만큼, 개인에게 맞춤화된 교육을 제공하기 위한 핵심 기술로 평가되고 있습니다. 

FeedGate팀은 Iscream 데이터셋을 이용하여 DKT모델을 구축할 예정입니다. 학생 개개인이 푼 문제 리스트와 정답 여부가 담긴 데이터를 바탕으로 이후에 제시된 문제를 맞출지 틀릴지를 예측하는 것을 목표로 삼고 있습니다.

![test](https://user-images.githubusercontent.com/46434838/119333156-82dc0000-bcc4-11eb-8074-3c4833ef308c.png)


# File Structure
## | Baseline
```
code/
│
├── dkt
│   ├── criterion.py
│   ├── dataloader.py
│   ├── metric.py
│   ├── model.py
│   ├── optimizer.py
│   ├── scheduler.py
│   ├── trainer.py
│   └── utils.py
├── config
│   ├── config.py
│   └── cofing.yml
├── asset
│   ├── assessmentItemID_classes.npy
│   ├── KnowledgeTag_classes.npy
│   └── testId_classes.nmp
├── .gitignore
├── args.py
├── baseline.ipynb
├── inference.py
├── README.md
├── requirements.txt
└── train.py
```

## | Input
Input directory는 로컬 작업 환경에만 존재합니다.
```
input/
│ 
└── data/
    └── train_dataset/
        ├── sample_submission.csv
        ├── test_data.csv
        └── train_data.csv
```
