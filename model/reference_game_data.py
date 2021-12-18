from torch.utils.data import Dataset
from pycocotools.coco import COCO
import pandas as pd
from tqdm import tqdm
import random

class ReferenceGameData(Dataset):
    
    def __init__(self, split, num_images, num_samples):
        file = open('fairseq-image-captioning/splits/karpathy_'+split+'_images.txt', 'r')
        self.images = [image.split(' ')[0] for image in file.readlines()]
        self.caps = [COCO('fairseq-image-captioning/mscoco/annotations/captions_val2014.json'),
                     COCO('fairseq-image-captioning/mscoco/annotations/captions_train2014.json')]
        self.num_images = num_images
        self.num_samples = num_samples
        
    def __getitem__(self, index):
        images = random.sample(self.images, self.num_images)
        captions = []
        for image in images:
            image_captions = self.caps[0].loadAnns(self.caps[0].getAnnIds(imgIds=int(image[-10:-4]))) + \
                             self.caps[1].loadAnns(self.caps[1].getAnnIds(imgIds=int(image[-10:-4])))
            captions.append(random.choice(image_captions)['caption'])
        return {'images':images, 'captions':captions}
        
    def __len__ (self):
        return self.num_samples
        
        
class ImageData(Dataset):
    
    def __init__(self, split, model, question_type = 'polar'):
        self.question_type = question_type
        data_file = 'outputs/image_question_data.json' if question_type == 'polar' else 'outputs/what_image_question_data.json'
        
        try:
            data = pd.read_json(data_file)
            self.images = list(data['images'])
            self.questions = list(data['questions'])
        except:
            print('Initializing training data...')
            self.images = []
            self.questions = []
            self.responses = []
            for image in tqdm(open('fairseq-image-captioning/splits/karpathy_'+split+'_images.txt', 
                              'r').readlines()):
                image = image.split(' ')[0]
                for question in model.get_questions([image],question_type=self.question_type)[0]:
                    self.images.append(image)
                    if self.question_type == 'polar':
                        self.questions.append(question)
                    elif self.question_type == 'what':
                        self.questions.append([question[0], 
                                               model.words_to_ids(question[1])[1].item()])
                    else:
                        raise ValueError(self.question_type+' is not a vlaid question type')
            pd.DataFrame({'images':self.images,
                          'questions':self.questions}).to_json(data_file)

    def __getitem__(self, index):
        image = self.images[index]
        if self.question_type == 'polar':
            response = random.choice([0,1])
            if response == 0:
                question = random.choice(self.questions)
            else:
                question = self.questions[index]
        elif self.question_type == 'what':
            question, response = self.questions[index]
            
        return {'image': image, 'question': question, 'response': response}
        
    def __len__ (self):
        return len(self.images)