from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from torch.utils.data import Dataset
import numpy as np
import os
import os.path
import torch
from torch.utils.data import DataLoader
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import sys
import time

DATA_PATH = 'clsapluscodes.txt'
MODEL = 'gpt2-medium'
EXP_NAME = 'robocanon1'

#Creates pairs of sequences (X,Y) XY is a coherent string
class LMData(Dataset):
    def __init__(self,textPath,tokenizer,kernel_size=256,stride=128):
        self.kernel_size = kernel_size
        self.stride = stride
        with open(textPath) as x: self.text = x.read()
        self.tokens = []
        for i in range(self.get_length(self.text,self.kernel_size,self.kernel_size)):
            cur_text = self.get_independent_window(self.text,i,self.kernel_size,self.kernel_size)
            cur_tokens = tokenizer(cur_text)
            self.tokens.extend(cur_tokens['input_ids'])
        self.tokens = np.array(self.tokens)
        
    def get_independent_window(self,iterable,i,kernel_size,stride):
        return iterable[stride*i:(stride*i+kernel_size)]
        
    def get_length(self,iterable,kernel_size,stride):
        return ((len(iterable) - kernel_size)// stride)
        
    
    def __getitem__(self,i):
        cur_tokens = self.get_independent_window(self.tokens,i,self.kernel_size+1,self.stride)
        X = cur_tokens[:self.kernel_size]
        Y = cur_tokens[1:(self.kernel_size+1)]
        return(X,Y)
    
    def __len__(self):
        #Usually + 1, but we want to generate two sequences every time!
        return self.get_length(self.tokens,self.kernel_size+1,self.stride)



def _init_(exp_name):
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+ exp_name):
        os.makedirs('checkpoints/'+ exp_name)
    if not os.path.exists('checkpoints/'+ exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+ exp_name+'/'+'models')

def train(exp_name):
    train_dataset = LMData(DATA_PATH,tokenizer)
    train_loader = DataLoader(train_dataset,num_workers=8,batch_size=2, shuffle=True, drop_last=True)

    #model = DistilBertForMaskedLM.from_pretrained(MODEL).cuda()
    device = 0
    
    sys.stdout.flush()
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    
    LR = 0.0001
    MOMENTUM = 0.9
    EPOCHS = 100

    #opt = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=1e-4)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(opt, EPOCHS, eta_min=LR)
    
    #criterion = cal_loss
    scaler = torch.cuda.amp.GradScaler()

    best_loss = 100.0
    for epoch in range(EPOCHS):
        t = time.time()

        losses = []
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        for before, after in train_loader:
            before, after = before.to(device), after.to(device)
            
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                output = model(input_ids=before,labels=after)
                logits = output['logits']
                loss = output['loss']
            losses.append(loss.detach().cpu().numpy())
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            #opt.step()
            #break;
            
            
            count += 1
            if count % 1000 == 0:
                print('Epoch {0} Batch {1} Loss: {2}'.format(epoch,count,np.mean(losses)))
                sys.stdout.flush()
        cur_loss =np.mean(losses)
        print('Epoch {0} Overall Loss: {1}'.format(epoch,cur_loss))
        sys.stdout.flush()
            
        scheduler.step()

        if best_loss >= cur_loss:
            best_loss = cur_loss
            torch.save(model.state_dict(), 'checkpoints/%s/models/model.t7' % exp_name)
            
        #VALIDATE
        with torch.no_grad():
            model.eval()
            for i in range(3):
                #outputs = model.generate('In ',do_sample=True, max_length=60, pad_token_id=50256)
                print(generator('In ',pad_token_id=50256, num_return_sequences=3,max_length=50))
                #print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
        #break
        print('Epoch {0} Time: {1}'.format(epoch,time.time() - t))

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL).cuda()
generator = pipeline(task="text-generation", model=model, tokenizer=tokenizer,device=0)
_init_(EXP_NAME)
train(EXP_NAME)