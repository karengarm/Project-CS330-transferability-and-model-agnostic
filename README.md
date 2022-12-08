# Exploring transferability and model agnostic meta learning across NLP Tasks

Code for our projec "Exploring transferability and model agnostic meta learning across NLP Tasks". CS330 Deep Multi-Task and Meta Learning, Stanford University.

![plot](Poster_picture.PNG)

## Fine-tuning GPT2 on downstream NLP tasks

For each (intermediate question answering tasks, text classification target task) group, we use a pretrained GPT2 model, fine-tune it sequentally on the intermediate tasks, and then fine-tune the resulting model on the text classification target task. We have separate code for fine-tuning GPT2 on each class of problems, including the intermediate task question answering and the target task text classification/regression. 

The following example code fine-tunes GPT2 on the datasets squad, squadv2, duorc-p as intermediate tasks. The fine-tuned intermediate model then serves as
starting point to fine-tune on the target text classification task:

``` 
!python run_intermediate_source_GPT2.py --dataset_name 'squad' --output_dir './pretrained_model_squad'
!python run_intermediate_source_GPT2.py --dataset_name 'squad_v2' --model_name './pretrained_model_squad'  --output_dir './pretrained_model_squad_v2'
!python run_intermediate_source_GPT2.py --dataset_name 'duorc-p' --model_name './pretrained_model_squad_v2'  --output_dir './pretrained_model_duorc-p'
!python run_target_task_GPT2_sst2_multiple.py --pretrained_model './pretrained_model_duorc-p'
```





