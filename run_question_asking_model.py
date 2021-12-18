from model.question_asking_model import QuestionAskingModel
from model.reference_game_data import ReferenceGameData

import torch
from torch.utils.data import Dataset, DataLoader

import argparse
import random
from tqdm import tqdm

random.seed(123)

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--include_what', action='store_true')
parser.add_argument('--target_idx', type=int, default=0)
parser.add_argument('--max_num_questions', type=int, default=25)
parser.add_argument('--num_images', type=int, default=10)
parser.add_argument('--description_given', action='store_true')
parser.add_argument('--beam', type=int, default=1)
args = parser.parse_args()

args.load_response_model = True

results = {'images':[], 'target':[], 'description':[], 'question':[], 'response':[],
           'p_y_x':[], 'p_r_qy':[], 'p_y_xqr':[]}

model = QuestionAskingModel(args)
dataloader = DataLoader(dataset=ReferenceGameData(split='test', 
                                                  num_images=args.num_images, 
                                                  num_samples=1000))

for batch in tqdm(dataloader):
    images = [image[0] for image in batch['images'][:args.num_images]]
    image_embeddings = model.get_image_embeddings(images)
    
    questions, target_questions = model.get_questions(images, args.target_idx)
    if args.include_what:
        what_questions, target_what_questions = model.get_questions(images, args.target_idx, question_type='what')
        questions += what_questions
        target_questions += target_what_questions
    question_embeddings = model.get_question_embeddings(questions)
    
    if args.description_given:
        description = batch['captions'][args.target_idx][0]
        p_y_x = model.get_image_prob([description],images)[0]
    else:
        description = None
        p_y_x = (torch.ones(args.num_images)/args.num_images).to(model.device)

    question_set = questions.copy()
    for j in range(min(args.max_num_questions, len(question_set))):
        question = model.select_best_question(p_y_x, question_set, question_embeddings, image_embeddings)
        
        if type(question) == tuple:
            question, response = question
            response_id = [model.words_to_ids(response)[1].item()]
            p_r_qy = model.response_likelihood(None, image_embeddings, 
                                              question_embeddings[questions.index(question)], 
                                              question_type='what')
            p_r_qy = p_r_qy[:,model.words_to_ids(response)[1].item()]
        else:
            response = 'yes' if question in target_questions else 'no'
            p_r_qy = model.response_likelihood(response, image_embeddings, 
                                              question_embeddings[questions.index(question)])

        p_y_xqr = p_y_x*p_r_qy
        p_y_xqr = p_y_xqr/torch.sum(p_y_xqr)if torch.sum(p_y_xqr) != 0 else torch.zeros_like(p_y_xqr)

        results['images'].append(images)
        results['target'].append(images[args.target_idx])
        results['description'].append(description)
        results['question'].append(question)
        results['response'].append(response)
        results['p_y_x'].append(p_y_x.tolist())
        results['p_r_qy'].append(p_r_qy.tolist())
        results['p_y_xqr'].append(p_y_xqr.tolist())
        
        p_y_x = p_y_xqr
    
results_df = pd.DataFrame(results)
filename = 'outputs/'+args.dataset+'_'+str(args.num_images)+'_polar'
if args.description_given:
    filename += '_description_given'
if arg.sinclude_what:
    filename = filename.replace('polar','what+polar')
filename += '.json'
results_df.to_json(filename)