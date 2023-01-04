#coding: UTF-8
import time
import torch
import models
from config import opt
import data

from tqdm import tqdm

from torch.utils.data import DataLoader
import torch.nn.functional as F

t = 0

def train(**kwargs):
    global t

    opt._parse(kwargs)

    model_index = []
    optimizer_index = []
    criterion_index = []
    adj_index = []
    features_index = []
    labels_index = []
    idx_train_index = []
    idx_val_index = []
    idx_test_index = []

    adj = 0

    lr = opt.lr
    NC = opt.NC

    data.load_data(0, 0, opt)
    for calculator_index in range(NC):
        adj_now, features_now, labels_now, idx_train_now, idx_val_now, idx_test_now, adj = data.load_data(calculator_index, NC, opt)
        if opt.model == 'PyGCN':
            model_now = getattr(models, opt.model)(features_now.shape[1], 128, max(labels_now) + 1).train()
        elif opt.model == 'PyGAT':
            model_now = getattr(models, opt.model)(features_now.shape[1], 8, max(labels_now) + 1, dropout=0.6, alpha=0.2, nheads=8).train()
        elif opt.model == 'PyGraphsage':
            model_now = getattr(models, opt.model)(features_now.shape[1], 8, max(labels_now) + 1)
        model_now = model_now.to(opt.device)
        adj_now = adj_now.to(opt.device)
        features_now = features_now.to(opt.device)

        train_now = data.Dataload(labels_now, idx_train_now)
        val_now = data.Dataload(labels_now, idx_val_now)
        test_now = data.Dataload(labels_now, idx_test_now)

        train_dataloader = DataLoader(train_now, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
        val_dataloader = DataLoader(val_now, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
        test_dataloader = DataLoader(test_now, opt.batch_size, shuffle=True, num_workers=opt.num_workers)

        criterion_now = F.nll_loss
        optimizer_now = torch.optim.Adam(model_now.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
        lr = opt.lr
        model_index.append(model_now)
        adj_index.append(adj_now)
        features_index.append(features_now)
        criterion_index.append(criterion_now)
        optimizer_index.append(optimizer_now)
        labels_index.append(labels_now)
        idx_train_index.append(idx_train_now)
        idx_val_index.append(idx_val_now)
        idx_test_index.append(idx_test_now)

    if opt.model == 'PyGCN':
        for epoch in range(opt.max_epoch):

            for trains, labels in tqdm(train_dataloader):

                labels = labels.to(opt.device)
                trains = trains.to(opt.device)

                weight_1 = 0
                weight_2 = 0
                bias_1 = 0
                bias_2 = 0

                for calculator_index in range(NC):
                    # global t
                    t = time.time()

                    model_now = model_index[calculator_index]
                    adj_now = adj_index[calculator_index]
                    features_now = features_index[calculator_index]
                    criterion_now = criterion_index[calculator_index]
                    optimizer_now = optimizer_index[calculator_index]
                    labels_now = labels_index[calculator_index]
                    idx_train_now = idx_train_index[calculator_index]
                    idx_val_now = idx_val_index[calculator_index]
                    idx_test_now = idx_test_index[calculator_index]


                    optimizer_now.zero_grad()
                    outputs_now = model_now(features_now, adj_now)
                    # loss_now = criterion_now(outputs_now[idx_train_now], torch.LongTensor(labels_now[idx_train_now]).to(opt.device))
                    loss_now = criterion_now(outputs_now[trains], labels)
                    loss_now.backward()
                    optimizer_now.step()
                    lr = lr * opt.lr_decay
                    for param_groups in optimizer_now.param_groups:
                        param_groups['lr'] = lr

                    model_now.train()
                    # model_now.save()

                    if calculator_index == 0:
                        weight_1 = model_now.gc1.weight.data
                        weight_2 = model_now.gc2.weight.data
                        bias_1 = model_now.gc1.bias.data
                        bias_2 = model_now.gc2.bias.data
                    else:
                        weight_1 = weight_1 + model_now.gc1.weight.data
                        weight_2 = weight_2 + model_now.gc2.weight.data
                        bias_1 = bias_1 + model_now.gc1.bias.data
                        bias_2 = bias_2 + model_now.gc2.bias.data
                
                weight_1 = weight_1 / NC
                weight_2 = weight_2 / NC
                bias_1 = bias_1 / NC
                bias_2 = bias_2 / NC

                for calculator_index in range(NC):
                    model_now = model_index[calculator_index]
                    model_now.gc1.weight.data = weight_1
                    model_now.gc2.weight.data = weight_2
                    model_now.gc1.bias.data = bias_1
                    model_now.gc2.bias.data = bias_2
                    model_index[calculator_index] = model_now
            

            model_now = model_index[0]
            idx_val_now = idx_val_index[0]
            idx_test_now = idx_test_index[0]
            features_now = features_index[0]
            labels_now = labels_index[0]
            
            # evalute(opt, model_now, idx_val_now, epoch, features_now, adj, labels_now)
            evalute(opt, model_now, idx_test_now, epoch, features_now, adj.to(opt.device), labels_now)
    
    elif opt.model == 'PyGAT':
        for epoch in range(opt.max_epoch):
            for trains, labels in tqdm(train_dataloader):

                labels = labels.to(opt.device)
                trains = trains.to(opt.device)

                att_0_W = 0
                att_0_a = 0

                att_1_W = 0
                att_1_a = 0

                att_2_W = 0
                att_2_a = 0

                att_3_W = 0
                att_3_a = 0

                att_4_W = 0
                att_4_a = 0

                att_5_W = 0
                att_5_a = 0

                att_6_W = 0
                att_6_a = 0

                att_7_W = 0
                att_7_a = 0

                out_att_W = 0
                out_att_a = 0

                for calculator_index in range(NC):
                    # global t
                    t = time.time()

                    model_now = model_index[calculator_index]
                    adj_now = adj_index[calculator_index]
                    features_now = features_index[calculator_index]
                    criterion_now = criterion_index[calculator_index]
                    optimizer_now = optimizer_index[calculator_index]
                    labels_now = labels_index[calculator_index]
                    idx_train_now = idx_train_index[calculator_index]
                    idx_val_now = idx_val_index[calculator_index]
                    idx_test_now = idx_test_index[calculator_index]


                    optimizer_now.zero_grad()
                    outputs_now = model_now(features_now, adj_now)
                    # loss_now = criterion_now(outputs_now[idx_train_now], torch.LongTensor(labels_now[idx_train_now]).to(opt.device))
                    loss_now = criterion_now(outputs_now[trains], labels)
                    loss_now.backward()
                    optimizer_now.step()
                    lr = lr * opt.lr_decay
                    for param_groups in optimizer_now.param_groups:
                        param_groups['lr'] = lr

                    model_now.train()
                    # model_now.save()

                    if calculator_index == 0:
                        att_0_W = model_now.attention_0.W.data
                        att_0_a = model_now.attention_0.a.data

                        att_1_W = model_now.attention_1.W.data
                        att_1_a = model_now.attention_1.a.data

                        att_2_W = model_now.attention_2.W.data
                        att_2_a = model_now.attention_2.a.data

                        att_3_W = model_now.attention_3.W.data
                        att_3_a = model_now.attention_3.a.data

                        att_4_W = model_now.attention_4.W.data
                        att_4_a = model_now.attention_4.a.data

                        att_5_W = model_now.attention_5.W.data
                        att_5_a = model_now.attention_5.a.data

                        att_6_W = model_now.attention_6.W.data
                        att_6_a = model_now.attention_6.a.data

                        att_7_W = model_now.attention_7.W.data
                        att_7_a = model_now.attention_7.a.data
                        
                        out_att_W = model_now.out_att.W.data
                        out_att_a = model_now.out_att.a.data
                    else:
                        att_0_W = att_0_W + model_now.attention_0.W.data
                        att_0_a = att_0_a + model_now.attention_0.a.data

                        att_1_W = att_1_W + model_now.attention_1.W.data
                        att_1_a = att_1_a + model_now.attention_1.a.data

                        att_2_W = att_2_W + model_now.attention_2.W.data
                        att_2_a = att_2_a + model_now.attention_2.a.data

                        att_3_W = att_3_W + model_now.attention_3.W.data
                        att_3_a = att_3_a + model_now.attention_3.a.data

                        att_4_W = att_4_W + model_now.attention_4.W.data
                        att_4_a = att_4_a + model_now.attention_4.a.data

                        att_5_W = att_5_W + model_now.attention_5.W.data
                        att_5_a = att_5_a + model_now.attention_5.a.data

                        att_6_W = att_6_W + model_now.attention_6.W.data
                        att_6_a = att_6_a + model_now.attention_6.a.data

                        att_7_W = att_7_W + model_now.attention_7.W.data
                        att_7_a = att_7_a + model_now.attention_7.a.data
                        
                        out_att_W = out_att_W + model_now.out_att.W.data
                        out_att_a = out_att_a + model_now.out_att.a.data
                
                att_0_W = att_0_W / NC
                att_0_a = att_0_a / NC

                att_1_W = att_1_W / NC
                att_1_a = att_1_a / NC

                att_2_W = att_2_W / NC
                att_2_a = att_2_a / NC

                att_3_W = att_3_W / NC
                att_3_a = att_3_a / NC

                att_4_W = att_4_W / NC
                att_4_a = att_4_a / NC

                att_5_W = att_5_W / NC
                att_5_a = att_5_a / NC

                att_6_W = att_6_W / NC
                att_6_a = att_6_a / NC

                att_7_W = att_7_W / NC
                att_7_a = att_7_a / NC
                
                out_att_W = out_att_W / NC
                out_att_a = out_att_a / NC

                for calculator_index in range(NC):
                    model_now = model_index[calculator_index]
                    model_now.attention_0.W.data = att_0_W
                    model_now.attention_0.a.data = att_0_a

                    model_now.attention_1.W.data = att_1_W
                    model_now.attention_1.a.data = att_1_a

                    model_now.attention_2.W.data = att_2_W
                    model_now.attention_2.a.data = att_2_a

                    model_now.attention_3.W.data = att_3_W
                    model_now.attention_3.a.data = att_3_a

                    model_now.attention_4.W.data = att_4_W
                    model_now.attention_4.a.data = att_4_a

                    model_now.attention_5.W.data = att_5_W
                    model_now.attention_5.a.data = att_5_a

                    model_now.attention_6.W.data = att_6_W
                    model_now.attention_6.a.data = att_6_a

                    model_now.attention_7.W.data = att_7_W
                    model_now.attention_7.a.data = att_7_a
                    
                    model_now.out_att.W.data = out_att_W
                    model_now.out_att.a.data = out_att_a
                    model_index[calculator_index] = model_now
            

            model_now = model_index[0]
            idx_val_now = idx_val_index[0]
            idx_test_now = idx_test_index[0]
            features_now = features_index[0]
            labels_now = labels_index[0]
            
            # evalute(opt, model_now, idx_val_now, epoch, features_now, adj, labels_now)
            evalute(opt, model_now, idx_test_now, epoch, features_now, adj.to(opt.device), labels_now)

    elif opt.model == 'PyGraphsage':
        for epoch in range(opt.max_epoch):
            for trains, labels in tqdm(train_dataloader):

                labels = labels.to(opt.device)
                trains = trains.to(opt.device)

                att_bias = 0
                att_weight = 0

                sage1_W = 0
                sage1_bias = 0

                sage2_W = 0
                sage2_bias = 0

                for calculator_index in range(NC):
                    # global t
                    t = time.time()

                    model_now = model_index[calculator_index]
                    adj_now = adj_index[calculator_index]
                    features_now = features_index[calculator_index]
                    criterion_now = criterion_index[calculator_index]
                    optimizer_now = optimizer_index[calculator_index]
                    labels_now = labels_index[calculator_index]
                    idx_train_now = idx_train_index[calculator_index]
                    idx_val_now = idx_val_index[calculator_index]
                    idx_test_now = idx_test_index[calculator_index]


                    optimizer_now.zero_grad()
                    outputs_now = model_now(features_now, adj_now)
                    # loss_now = criterion_now(outputs_now[idx_train_now], torch.LongTensor(labels_now[idx_train_now]).to(opt.device))
                    loss_now = criterion_now(outputs_now[trains], labels)
                    loss_now.backward()
                    optimizer_now.step()
                    lr = lr * opt.lr_decay
                    for param_groups in optimizer_now.param_groups:
                        param_groups['lr'] = lr

                    model_now.train()
                    # model_now.save()

                    if calculator_index == 0:
                        att_bias = model_now.att.bias.data
                        att_weight = model_now.att.weight.data

                        sage1_W = model_now.sage1.W.data
                        sage1_bias = model_now.sage1.bias.data

                        sage2_W = model_now.sage2.W.data
                        sage2_bias = model_now.sage2.bias.data
                    else:
                        att_bias = att_bias + model_now.att.bias.data
                        att_weight = att_weight + model_now.att.weight.data

                        sage1_W = sage1_W + model_now.sage1.W.data
                        sage1_bias = sage1_bias + model_now.sage1.bias.data

                        sage2_W = sage2_W + model_now.sage2.W.data
                        sage2_bias = sage2_bias + model_now.sage2.bias.data
                
                att_bias = att_bias / NC
                att_weight = att_weight / NC

                sage1_W = sage1_W / NC
                sage1_bias = sage1_bias / NC

                sage2_W = sage2_W / NC
                sage2_bias = sage2_bias / NC

                for calculator_index in range(NC):
                    model_now = model_index[calculator_index]
                    model_now.att.bias.data = att_bias
                    model_now.att.weight.data = att_weight

                    model_now.sage1.W.data = sage1_W
                    model_now.sage1.bias.data = sage1_bias

                    model_now.sage2.W.data = sage2_W
                    model_now.sage2.bias.data = sage2_bias
                    model_index[calculator_index] = model_now
            

            model_now = model_index[0]
            idx_val_now = idx_val_index[0]
            idx_test_now = idx_test_index[0]
            features_now = features_index[0]
            labels_now = labels_index[0]
            
            # evalute(opt, model_now, idx_val_now, epoch, features_now, adj, labels_now)
            evalute(opt, model_now, idx_test_now, epoch, features_now, adj.to(opt.device), labels_now)



def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def evalute(opt, model, idx_val ,epoch, features, adj, labels):
    global t
    model.eval()
    critetion = F.nll_loss
    with torch.no_grad():
        outputs = model(features, adj)
        loss = critetion(outputs[idx_val], torch.LongTensor(labels[idx_val]).to(opt.device))
    acc = accuracy(outputs[idx_val], torch.LongTensor(labels[idx_val]))
    print(#'DEVICE{:04d}'.format(calculator_index + 1),
        'Epoch: %3s' % str(epoch),
        'loss: {:.4f}'.format(loss.item()),
        'acc: {:.4f}'.format(acc.item()),
        'time: {:.4f}s'.format(time.time() - t))
    return acc


if __name__ == '__main__':
    train()