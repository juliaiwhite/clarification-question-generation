from model.question_generator import QuestionGenerator
from model.response_model import ResponseModel

from fairseq import tasks, checkpoint_utils
from fairseq.data import encoders

import sys
sys.path.append('fairseq-image-captioning')
import task
import data

import torch
import torch.nn.functional as F

import numpy as np
import operator
import random

class QuestionAskingModel():
    def __init__(self, args):
        # Parameters
        self.device = args.device
        self.include_what = args.include_what
        self.beam = args.beam
        self.max_length = 128
        self.eps = 1e-25
        
        # Initialize question generation model
        class Namespace:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        model_args = Namespace(beam=self.beam, bpe='subword_nmt', bpe_codes='fairseq-image-captioning/output/codes.txt', bpe_separator='@@', buffer_size=0, captions_dir='fairseq-image-captioning/output/', captions_lang='en', cpu=False, criterion='cross_entropy', dataset_impl=None, decoding_format=None, diverse_beam_groups=-1, diverse_beam_strength=0.5, empty_cache_freq=0, features='obj', features_dir='output', force_anneal=None, fp16=False, fp16_init_scale=128, fp16_scale_tolerance=0.0, fp16_scale_window=None, gen_subset='test', input='demo/demo-ids.txt', iter_decode_eos_penalty=0.0, iter_decode_force_max_iter=False, iter_decode_max_iter=10, lenpen=1, log_format=None, log_interval=1000, lr_scheduler='fixed', lr_shrink=0.1, match_source_len=False, max_len_a=0, max_len_b=200, max_sentences=None, max_source_positions=64, max_target_positions=1024, max_tokens=None, memory_efficient_fp16=False, min_len=1, min_loss_scale=0.0001, model_overrides='{}', momentum=0.99, moses_no_dash_splits=False, moses_no_escape=False, moses_source_lang=None, moses_target_lang=None, nbest=1, no_beamable_mm=False, no_early_stop=False, no_progress_bar=False, no_repeat_ngram_size=0, num_shards=1, num_workers=1, optimizer='nag', path='fairseq-image-captioning/checkpoint24.pt', prefix_size=0, print_alignment=False, print_step=False, quiet=False, remove_bpe=None, replace_unk=None, required_batch_size_multiple=8, results_path=None, retain_iter_history=False, sacrebleu=False, sampling=False, sampling_topk=-1, sampling_topp=-1.0, score_reference=False, seed=1, shard_id=0, skip_invalid_size_inputs_valid_test=False, task='captioning', temperature=1.0, tensorboard_logdir='', threshold_loss_scale=None, tokenizer='moses', unkpen=0, unnormalized=False, user_dir='task', warmup_updates=0, weight_decay=0.0)

        task = tasks.setup_task(model_args)
        captions_dict = task.target_dictionary

        self.captioner_model = checkpoint_utils.load_model_ensemble(model_args.path.split(':'), task=task)[0][0]

        self.captioner_model.make_generation_fast_(
            beamable_mm_beam_size=None if model_args.no_beamable_mm else model_args.beam,
            need_attn=model_args.print_alignment,
        )
        self.captioner_model.to(self.device)

        self.tokenizer = encoders.build_tokenizer(model_args)
        self.bpe = encoders.build_bpe(model_args)
        self.vocab = task.target_dictionary
        
        self.question_generator = QuestionGenerator()
        
        # Initialize response model
        self.response_model = ResponseModel(out_channels = 1)
        if args.load_response_model:
            self.response_model.load_state_dict(torch.load('model/response_models/response_model.pth.tar',
                                                           map_location=self.device))
        self.response_model.to(self.device)
        if self.include_what:
            self.what_response_model = ResponseModel(out_channels = len(self.vocab))
            if args.load_response_model:
                self.what_response_model.load_state_dict(torch.load('model/response_models/what_response_model.pth.tar',
                                                                    map_location=self.device))
            self.what_response_model.to(self.device)
        
        # Initialize dataset
        def get_image_ds(split):
            try:
                return data.ObjectFeaturesDataset('fairseq-image-captioning/output/features-obj', 
                                              data.read_image_ids('fairseq-image-captioning/output/'+split+'-ids.txt'), 
                                              data.read_image_metadata('fairseq-image-captioning/output/features-obj/'+split+'-metadata.csv'))
            except:
                raise ValueError('rename metadata.csv to '+split+'-metadata.csv and move from fairseq-image-captioning/output/'+split+'features-obj to fairseq-image-captioning/output/features-obj')
        self.image_ds = get_image_ds('train')
        self.image_ds.image_metadata = {**self.image_ds.image_metadata, 
                                        **get_image_ds('valid').image_metadata, 
                                        **get_image_ds('test').image_metadata}

    def get_image_embeddings(self, images):
        image_embeddings = []
        for image in images:
            features, locations = self.image_ds.read_data(int(image[-10:-4]))
            image_embedding = self.captioner_model.encoder(features.unsqueeze(0).to(self.device), 
                                                 [features.shape[0]], 
                                                 locations.unsqueeze(0).to(self.device))[0].mean(0).detach()
            image_embeddings.append(image_embedding.flatten())
        return image_embeddings
    
    def words_to_ids(self, words):
        ids = [self.vocab.index(word_id) for word_id in self.bpe.encode(self.tokenizer.encode(words)).split(' ')]
        return torch.tensor([self.vocab.bos()] + ids + [self.vocab.eos()]).to(self.device)
        
    def ids_to_words(self, ids):
        return self.tokenizer.decode(self.bpe.decode(ids))
    
    def get_question_embeddings(self, questions):
        question_embeddings = []
        for question in questions:
            if type(question) == tuple:
                question = question[0]
            question_ids = self.words_to_ids(question)
            question_embedding = self.captioner_model.decoder.embed_tokens(question_ids).mean(0).detach()
            question_embeddings.append(question_embedding)
        return question_embeddings
    
    def get_questions(self, images, target_idx=0, return_target_questions=True, question_type='polar'):
        questions = []
        for i, image in enumerate(images):
            description = self.generate_description(image, images)
            
            if question_type == 'polar':
                image_questions = self.question_generator.generate_is_there_question(description)
            elif question_type == 'what':
                image_questions = self.question_generator.generate_what_question(description)
            else:
                raise ValueError(question_type+' is not a vlaid question type')
                
            if return_target_questions and i == target_idx: 
                target_questions = image_questions
            questions += image_questions
        questions = list(set(questions))
        random.shuffle(questions)
        if return_target_questions: 
            return questions, target_questions
        return questions
    
    def generate_description(self, image, images):
        features, locations = self.image_ds.read_data(int(image[-10:-4]))
        topk_batch, topk_scores = self.beam_sample(features.unsqueeze(0),
                                                   [features.shape[0]],
                                                   locations.unsqueeze(0))
        topk_batch_strings = []
        for caption in topk_batch:
            topk_batch_strings.append(self.tokenizer.decode(self.bpe.decode(self.vocab.string(caption))))
        
        image_prob = self.get_image_prob(topk_batch_strings, images)
        utility  = torch.index_select(image_prob, 1, torch.tensor(images.index(image)).to(self.device))
        
        reranked_descriptions = sorted(zip(list(utility.data.cpu().numpy()),
                            topk_batch_strings),
                        key = operator.itemgetter(0))[::-1]
        return reranked_descriptions[0][1]
    
    def get_image_prob(self, descriptions, images):
        description_ids = []
        for description in descriptions:
            description_ids.append(self.words_to_ids(description))
        description_likelihood = self.get_description_likelihood(description_ids, images)
        return torch.exp(description_likelihood)

    def get_description_likelihood(self, description_ids_list, images):
        likelihood = []
        for description_ids in description_ids_list:
            log_probs = []
            for image in images:
                features, locations = self.image_ds.read_data(int(image[-10:-4]))
                enc_out = self.captioner_model.encoder(features.unsqueeze(0).to(self.device), 
                                             [features.shape[0]], 
                                             locations.unsqueeze(0).to(self.device))
                dec_out, extra = self.captioner_model.decoder(description_ids.unsqueeze(0), 
                                                    encoder_out=enc_out)
                log_probs.append(self.captioner_model.get_normalized_probs((dec_out, extra), log_probs=True))
            log_probs = torch.cat(log_probs)

            log_likelihood = []
            for i in range(log_probs.shape[0]) :
                log_likelihood.append(-1 * F.cross_entropy(log_probs[i], description_ids, reduction="none", ignore_index=0))
            log_likelihood= torch.stack(log_likelihood, 0)
            likelihood.append(torch.sum(log_likelihood, dim = 1).unsqueeze(0))
        likelihood = torch.cat(likelihood)
        total = torch.logsumexp(likelihood, dim = 1)
        return likelihood - total.reshape((total.shape[0], 1))

    def response_likelihood(self, response, image_embeddings, question_embedding, question_type='polar'):
        joint_embeddings = []
        for image_embedding in image_embeddings:
            joint_embeddings.append(torch.cat([image_embedding.flatten(),
                                               question_embedding]).unsqueeze(0))
        joint_embeddings = torch.cat(joint_embeddings)
        with torch.no_grad():
            if question_type == 'polar':
                p_r_qy = torch.sigmoid(self.response_model(joint_embeddings)).squeeze()
            elif question_type == 'what':
                p_r_qy = torch.sigmoid(self.what_response_model(joint_embeddings)).squeeze()
            else:
                raise ValueError(question_type+' is not a vlaid question type')
                
        if response == 'no':
            return 1-p_r_qy
        return p_r_qy
        
    def select_best_question(self, p_y_x, question_set, question_embeddings, image_embeddings):
        H_y_rxq = [0]*len(question_set)
        for i, (question, question_embedding) in enumerate(zip(question_set,question_embeddings)):
            if type(question) == tuple:
                p_r_qy = self.response_likelihood(None, image_embeddings, question_embedding, question_type='what')
                p_r_qy = torch.cat([p_r_qy[:,response] for response in p_r_qy.argmax(1)])
            else:
                p_r_qy = self.response_likelihood(None, image_embeddings, question_embedding)
                p_r_qy = [p_r_qy, 1-p_r_qy]

            p_y_xqr = torch.stack([p_y_x*p_r_qy[r] for r in range(len(p_r_qy))])
            p_y_xqr = [p_y_xqr[r]/torch.sum(p_y_xqr[r]) if torch.sum(p_y_xqr[r]) != 0 \
                       else [0]*len(p_y_xqr[r]) for r in range(len(p_y_xqr))]

            H_y_rxq[i] = torch.sum(torch.stack([p_r_qy[r]*p_y_x*torch.log2(1/(p_y_xqr[r]+self.eps)) for r in range(len(p_r_qy))]))
           
        IG = - torch.stack(H_y_rxq).unsqueeze(1)
        ranked_questions = sorted(zip(list(IG.data.cpu().numpy()), question_set),
                                  key = operator.itemgetter(0))[::-1]
        return ranked_questions[0][1]
    
    
    def beam_sample(self, sample_features, sample_length, sample_location, sample_size=1, feed_cap=None):
        # this function is taken from generate.py in krasserm/fairseq-image-captioning (with minor changes)
        
        incremental_state = {}
        
        feed_tokens = torch.ones(sample_size, 1, dtype=torch.long).cuda() * self.vocab.eos() \
                      if feed_cap == None else [self.words_to_ids(feed_cap)[:-1]]
            
        enc_out = self.captioner_model.encoder(sample_features.cuda(), torch.tensor(sample_length).cuda(), sample_location.cuda())
        dec_out, extra = self.captioner_model.decoder(feed_tokens, encoder_out=enc_out, incremental_state=incremental_state)

        lprobs = self.captioner_model.get_normalized_probs((dec_out, extra), log_probs=True)
        lprobs, tokens = torch.topk(lprobs, k=self.beam, dim=2)
        lprobs = lprobs.transpose(2, 1)

        tokens = tokens.transpose(2, 1)
        tokens = torch.cat([feed_tokens.repeat(1, self.beam, 1), tokens], dim=2)
        
        mask = torch.zeros(sample_size, self.beam, len(feed_tokens[0])+2, dtype=torch.bool)
        mask[:, :, :] = True

        new_order = torch.tensor(np.repeat(range(sample_size), self.beam)).cuda()
        enc_out = self.captioner_model.encoder.reorder_encoder_out(enc_out, new_order)
        incremental_state = self.captioner_model.decoder.reorder_incremental_state(incremental_state, new_order)
        
        for _ in range(self.max_length):
            tokens_batch = tokens.flatten(0, 1)
            dec_out, extra = self.captioner_model.decoder(tokens_batch, encoder_out=enc_out, incremental_state=incremental_state)
            lprobs_batch = self.captioner_model.get_normalized_probs((dec_out, extra), log_probs=True)
            lprobs_batch = lprobs_batch[:, -1, :]
            lprobs_batch = lprobs_batch.reshape(tokens.shape[0], tokens.shape[1], -1)
            lprobs_k, tokens_k = torch.topk(lprobs_batch, k=self.beam, dim=2)

            tokens_repeated = torch.repeat_interleave(tokens, self.beam, dim=1)
            tokens_k_flattened = tokens_k.flatten().view(sample_size, -1, 1)
            tokens_cat = torch.cat([tokens_repeated, tokens_k_flattened], dim=2)

            mask_repeated = torch.repeat_interleave(mask, self.beam, dim=1).cuda()
            mask_k_flattened = (tokens_k_flattened != self.vocab.eos()) & mask_repeated[:, :, -1:]
            mask_cat = torch.cat([mask_repeated, mask_k_flattened], dim=2)

            lprobs_repeated = torch.repeat_interleave(lprobs, self.beam, dim=1)
            lprobs_k_flattened = lprobs_k.flatten().view(sample_size, -1, 1)
            lprobs_cat = torch.cat([lprobs_repeated, lprobs_k_flattened], dim=2)
            lprobs_cat_masked = lprobs_cat * mask_cat[:, :, len(feed_tokens[0]):-1]
            
            num_tokens = torch.sum(mask_cat[:, :, len(feed_tokens[0]):-1], dim=2)    
            scores = torch.sum(lprobs_cat_masked, dim=2)
            scores_mask = torch.zeros(sample_size, self.beam * self.beam, dtype=torch.bool)

            for i in range(sample_size):
                for j in range(self.beam):
                    first = j * self.beam
                    start = first + 1
                    end = first + self.beam
                    scores_mask[i, start:end] = torch.sum(mask_cat[i, first:end, -1]) == 0

            for i in range(sample_size):
                scores[i][scores_mask[i]] = -1e8

            top_values, top_indices = torch.topk(scores, k=self.beam)
            incremental_state = self.captioner_model.decoder.reorder_incremental_state(incremental_state=incremental_state, 
                                                                             new_order=top_indices.flatten() // self.beam)

            tokens_list = []
            lprobs_list = []
            mask_list = []
            
            for i in range(sample_size):
                tokens_selected = tokens_cat[i][top_indices[i]]
                tokens_list.append(tokens_selected)

                lprobs_selected = lprobs_cat[i][top_indices[i]]
                lprobs_list.append(lprobs_selected)

                mask_selected = mask_cat[i][top_indices[i]]
                mask_list.append(mask_selected)

            tokens = torch.stack(tokens_list, dim=0)
            lprobs = torch.stack(lprobs_list, dim=0)
            mask = torch.stack(mask_list, dim=0)

            if torch.sum(mask[:, :, -1]) == 0:
                break

        result_mask = mask[:, :, len(feed_tokens[0]):-1]
        result_tokens = tokens[:, :, len(feed_tokens[0]):] * result_mask
        result_lprobs = lprobs * result_mask
        
        result_num_tokens = torch.sum(result_mask, dim=2)
        result_scores = torch.sum(result_lprobs, dim=2)
        
        return result_tokens[0], result_scores