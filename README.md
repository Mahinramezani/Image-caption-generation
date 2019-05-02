# Image-caption-generation

Image Caption Generation is a task of generating the natural language description for the input image. This problem is interesting because it connects Computer Vision with Natural Language Processing which are two major fields in Artificial Intelligence. 


## dataset

The dataset can be downloaded by submitting the request form: https://forms.illinois.edu/sec/1713398


I used GloVe tool for embedding each word in data. So you need "glove.6B.200d.txt" file in order to train models. you can download it here: https://nlp.stanford.edu/projects/glove/ 


## how to run 

After downloading the data, you should first run "image_preprocess-vgg.py" to create "features.pkl" file.
Then you can run "run.ipynb" to generate caption for test images. By changing image_index, you can test different images.
