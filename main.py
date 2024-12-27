from __future__ import print_function

from tensorboardX import SummaryWriter
import torch
from config import Config as C
import os
from utils import build_loaders, build_model, train, evaluate, get_lr, save_checkpoint, \
                  count_parameters, set_random_seed,score_full
import warnings
warnings.filterwarnings('ignore')

def log_train(C, summary_writer, e, loss, lr, teacher_forcing_ratio, scores=None):
    summary_writer.add_scalar(C.tx_train_loss, loss['total'], e)
    summary_writer.add_scalar(C.tx_train_cross_entropy_loss, loss['cross_entropy'], e)
    summary_writer.add_scalar(C.tx_train_contrastive_attention_loss, loss['contrastive_attention'], e)
    summary_writer.add_scalar(C.tx_train_cls_loss, loss['emotion_cls'], e)
    summary_writer.add_scalar(C.tx_lr, lr, e)
    summary_writer.add_scalar(C.tx_teacher_forcing_ratio, teacher_forcing_ratio, e)
    # print("[TRAIN] loss: {} (= CE {} + CA {})".format(
    #     loss['total'], loss['cross_entropy'], loss['contrastive_attention']))
    if scores is not None:
      for metric in C.metrics_full:
          summary_writer.add_scalar("TRAIN SCORE/{}".format(metric), scores[metric], e)
      print("scores: {}".format(scores))


def log_val(C, summary_writer, e, loss, test_vid2GTs, test_vid2pred, vid2idx, scores=None):
    summary_writer.add_scalar(C.tx_val_loss, loss['total'], e)
    summary_writer.add_scalar(C.tx_val_cross_entropy_loss, loss['cross_entropy'], e)
    summary_writer.add_scalar(C.tx_val_contrastive_attention_loss, loss['contrastive_attention'], e)
    summary_writer.add_scalar(C.tx_val_cls_loss, loss['emotion_cls'], e)
    # print("[VAL] loss: {} (= CE {} + CA {})".format(
    #     loss['total'], loss['cross_entropy'], loss['contrastive_attention']))
    if scores is not None:
        for metric in C.metrics_full:
            summary_writer.add_scalar("VAL SCORE/{}".format(metric), scores[metric], e)
        print("scores: {}".format(scores))

def write_logs(C):
    log_path = os.path.join( C.log_dpath, "logs.csv")
    with open(log_path, 'w') as fout:
        fout.write("batchsize,lr,lam,beta,em_wei,filepath,prepath\n")
        #line = ', '.join([ str(C.batch_size),str(C.lr),str(C.CA_lambda),str(C.CA_beta),str(C.em_wei),C.file_path,C.pretrained_fpath])
        line = ', '.join([ str(C.batch_size),str(C.lr),str(C.CA_lambda),str(C.CA_beta),str(C.em_wei),C.file_path])
        fout.write("{}\n".format(line))

def get_teacher_forcing_ratio(max_teacher_forcing_ratio, min_teacher_forcing_ratio, epoch, max_epoch):
    x = 1 - float(epoch - 1) / (max_epoch - 1)
    a = max_teacher_forcing_ratio - min_teacher_forcing_ratio
    b = min_teacher_forcing_ratio
    return a * x + b

def get_em_wei(C, vocab):
    weights = torch.ones(vocab.n_vocabs)
    em_wei = weights + C.em_wei
    indexs = []
    #em_words = open('workspace/EmCap/EmotionEval/em_words.txt','r')
    em_words = open('dataset/179_words.txt',"r")
    for idx,ddd in enumerate(em_words):
        em_word = ddd.split()[0]
        index = vocab.word2idx[em_word]
        indexs.append(index)
    indexs.sort()
    indexs = torch.tensor(indexs)
    weights.scatter_(0, indexs, em_wei) 
    return weights

def log_result(C,best_val_b1,best_val_b2,best_val_b3,best_val_b4,best_val_meteor,best_val_rl,best_val_cider,best_val_acc_sw,best_val_acc_c,best_val_bfs,best_val_cfs):
    log_path = os.path.join( C.log_dpath, "logs.csv")
    with open(log_path, 'a') as fout:
        fout.write(str([best_val_b1,best_val_b2,best_val_b3,best_val_b4,best_val_meteor,best_val_rl,best_val_cider,best_val_acc_sw,best_val_acc_c,best_val_bfs,best_val_cfs])+"\n")


 
def main():
    set_random_seed(C.seed)
    device_id = 1
    sub_mode = False
    torch.cuda.set_device(device_id)
    summary_writer = SummaryWriter(C.log_dpath)
    write_logs(C)
    if not os.path.exists(C.result_dir):
        os.makedirs(C.result_dir)
    
    train_iter, val_iter, test_iter, em_train_iter, em_test_iter, vocab = build_loaders(C)
    dataset = C.file_path
    model = build_model(C, vocab)
    print("#params: ", count_parameters(model))
    model = model.cuda()
    best_val_b4 = { 'b4': -1,"epoch":-1}
    best_val_b3 = { 'b3': -1,"epoch":-1}
    best_val_b2 = { 'b2': -1,"epoch":-1}
    best_val_b1 = { 'b1': -1,"epoch":-1}
    best_val_rl = { 'rl': -1,"epoch":-1}
    best_val_meteor = { 'meteor': -1,"epoch":-1}
    best_val_cider = { 'cider': -1,"epoch":-1}
    best_val_acc_sw = { 'acc_sw': -1,"epoch":-1}
    best_val_acc_c = { 'acc_c': -1,"epoch":-1}
    best_val_bfs = { 'BFS': -1,"epoch":-1}
    best_val_cfs = { 'CFS': -1,"epoch":-1}
    cross_wei = get_em_wei(C,vocab)
    cross_wei = cross_wei.cuda()

    #print("Pretrained decoder is loading from {}".format(C.pretrained_fpath))    
    #model.load_state_dict(torch.load(C.pretrained_fpath),strict=False)
    #scores, refs, hypos, vid2idx = score_full(model,em_test_iter,vocab,dataset)

    optimizer = torch.optim.Adamax(model.parameters(), lr=C.lr, weight_decay=1e-5)

    for e in range(1, C.epochs + 1):

        print()
        ckpt_fpath = C.ckpt_fpath_tpl_em.format(e)

        """ Train """
        teacher_forcing_ratio = get_teacher_forcing_ratio(C.decoder.max_teacher_forcing_ratio,
                                                        C.decoder.min_teacher_forcing_ratio,
                                                        e, C.epochs)
        train_loss = train(e, model, optimizer, em_train_iter, vocab, teacher_forcing_ratio,
                        C, cross_wei,sub_mode)
        log_train(C, summary_writer, e, train_loss, get_lr(optimizer), teacher_forcing_ratio)
        scores, refs, hypos, vid2idx = score_full(model,em_test_iter,vocab,dataset,sub_mode=sub_mode,e=e)
        if best_val_cfs["CFS"] < scores["CFS"]:
            best_val_cfs['CFS'] = scores['CFS']
            best_val_cfs['epoch'] = e
        if best_val_bfs["BFS"] < scores["BFS"]:
            best_val_bfs['BFS'] = scores['BFS']
            best_val_bfs['epoch'] = e
        if best_val_acc_c["acc_c"] < scores["Acc_c"]:
            best_val_acc_c['acc_c'] = scores['Acc_c']
            best_val_acc_c['epoch'] = e
        if best_val_acc_sw["acc_sw"] < scores["Acc_sw"]:
            best_val_acc_sw['acc_sw'] = scores['Acc_sw']
            best_val_acc_sw['epoch'] = e
        if best_val_cider["cider"] < scores["CIDEr"]:
            best_val_cider['cider'] = scores['CIDEr']
            best_val_cider['epoch'] = e
        if best_val_rl["rl"] < scores["ROUGE_L"]:
            best_val_rl['rl'] = scores['ROUGE_L']
            best_val_rl['epoch'] = e
        if best_val_meteor["meteor"] < scores["Meteor"]:
            best_val_meteor["meteor"] = scores["Meteor"]
            best_val_meteor['epoch'] = e
        if best_val_b4["b4"] < scores["Bleu_4"]:
            best_val_b4['b4'] = scores['Bleu_4']
            best_val_b4['epoch'] = e
        if best_val_b3["b3"] < scores["Bleu_3"]:
            best_val_b3['b3'] = scores['Bleu_3']
            best_val_b3['epoch'] = e
        if best_val_b2["b2"] < scores["Bleu_2"]:
            best_val_b2['b2'] = scores['Bleu_2']
            best_val_b2['epoch'] = e
        if best_val_b1["b1"] < scores["Bleu_1"]:
            best_val_b1['b1'] = scores['Bleu_1']
            best_val_b1['epoch'] = e
        print("Saving checkpoint at epoch={} to {}".format(e, ckpt_fpath))
        save_checkpoint(ckpt_fpath, e, model, optimizer)
        print(best_val_b1,best_val_b2,best_val_b3,best_val_b4,best_val_meteor,best_val_rl,best_val_cider,best_val_acc_sw,best_val_acc_c,best_val_bfs,best_val_cfs)
        log_result(C,best_val_b1,best_val_b2,best_val_b3,best_val_b4,best_val_meteor,best_val_rl,best_val_cider,best_val_acc_sw,best_val_acc_c,best_val_bfs,best_val_cfs)

main()