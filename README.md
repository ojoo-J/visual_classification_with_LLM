# visual_classification_with_LLM

## Configuration
You can change your configuration (hparams) in ```load.py```

e.g.)
- hparams['dataset] = 'imagenet'
- hparams['model_size'] = "ViT-B/32"
- hparams['gpt_model'] = 'gpt-4o'
- hparams['descriptor_fname'] = 'descriptors_imagenet_with_related_words'
- IMAGENET_DIR = '/data2/youngju/clip_classifier/ImageNet/'

...


## Generate the Descriptors
```
### first, fill in the blank: {your_key} (openai key)

python generate_descriptors.py
```

## Quantitative Evaluation (Accuracy)
```
python run1_evaluate_accuracy.py
```

## Expand to out-of-distribution dataset (ImageNet-A)
```
python run2_evaluate_with_ood_data.py
```
