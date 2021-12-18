from model.response_model import ResponseModel
from model.question_asking_model import QuestionAskingModel
from model.reference_game_data import ImageData

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

import argparse
import random
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--include_what', action='store_true')
parser.add_argument('--beam', type=int, default=1)
args = parser.parse_args()

args.load_response_model = False

model = QuestionAskingModel(args)


def get_acc(y_pred, y_test):
    y_pred_tag = y_pred.argmax(1) if args.include_what else \
                 torch.round(y_pred)
    return torch.sum(y_pred_tag == y_test)/len(y_test)
        
random.seed(123)

question_type = 'what' if args.include_what else 'polar'
trainloader = DataLoader(dataset=ImageData(split='train', model=model, question_type=question_type),batch_size=4)
validloader = DataLoader(dataset=ImageData(split='valid', model=model, question_type=question_type),batch_size=4)

get_loss = nn.CrossEntropyLoss() if args.include_what else nn.BCELoss()

if args.include_what:
    optimizer = torch.optim.Adam(model.what_response_model.parameters(), lr=0.00001)
else:
    optimizer = torch.optim.Adam(model.response_model.parameters(), lr=0.00001)
    
metrics = {'split':[],'epoch':[],'accuracy':[],'loss':[]}
best_loss = 100
epoch = 0
checkpoints_without_decrease_loss = 0
while checkpoints_without_decrease_loss < 4:
    if args.include_what:
        model.what_response_model.train()
    else:
        model.response_model.train()
    pbar = tqdm(trainloader)
    for i,batch in enumerate(pbar):
        images = batch['image']
        questions = batch['question']
        response = batch['response'].to(model.device)
        
        image_embedding = torch.stack(model.get_image_embeddings(images))
        question_embedding = torch.stack(model.get_question_embeddings(questions))
        inputs = torch.cat([image_embedding, question_embedding],dim=1)
        if args.include_what:
            output = torch.sigmoid(model.what_response_model(inputs)).squeeze()
        else:
            output = torch.sigmoid(model.response_model(inputs)).squeeze()
            response = response.float()
        
        acc = get_acc(output, response)
        loss = get_loss(output, response)
        loss.backward()
        optimizer.step()
        
        metrics['split'].append('train')
        metrics['epoch'].append(epoch)
        metrics['accuracy'].append(acc.item())
        metrics['loss'].append(loss.item())
        pbar.set_description(f'Loss: {loss.item():.5f} | Acc: {acc.item():.3f}')

        if i%1000 == 0:
            val_loss = []
            if args.include_what:
                model.what_response_model.eval()
            else:
                model.response_model.eval()
            with torch.no_grad():
                for batch in tqdm(trainloader):
                    images = batch['image']
                    questions = batch['question']
                    response = batch['response'].to(model.device)

                    image_embedding = torch.stack(model.get_image_embeddings(images))
                    question_embedding = torch.stack(model.get_question_embeddings(questions))
                    inputs = torch.cat([image_embedding, question_embedding],dim=1)
                    if args.include_what:
                        output = torch.sigmoid(model.what_response_model(inputs)).squeeze()
                    else:
                        output = torch.sigmoid(model.response_model(inputs)).squeeze()
                        response = response.float()
        
                    acc = get_acc(output, response)
                    loss = get_loss(output, response)

                    metrics['split'].append('train')
                    metrics['epoch'].append(epoch)
                    metrics['accuracy'].append(acc.item())
                    metrics['loss'].append(loss.item())
                    val_loss.append(loss.item())

            val_loss = torch.mean(torch.tensor(val_loss))
            print('Epoch ' + str(epoch) + ' - Val Loss: ' + str(val_loss))
            if val_loss < best_loss:
                if args.include_what:
                    torch.save(model.what_response_model.state_dict(), 'model/response_models/what_response_model.pth.tar')
                else:
                    torch.save(model.response_model.state_dict(), 'model/response_models/response_model.pth.tar')
                best_loss = val_loss
                checkpoints_without_decrease_loss = 0
            else:
                checkpoints_without_decrease_loss += 1
                print(str(checkpoints_without_decrease_loss) + ' checkpoint(s) without decreasing loss')
            epoch += 1