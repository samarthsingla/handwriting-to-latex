# %% [markdown]
# %pip install numpy==1.26.2

# %%
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

import logging

debug = logging.getLogger("Debug")
info  = print
plt.ion()   # interactive mode

# %%
#check GPU
device = None
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Running CUDA Mode:", device, torch.cuda.get_device_name(0))
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Running MPS Mode:", device)
else:
    device = torch.device("cpu")
    print("Running CPU Mode:", device)



# %% [markdown]
# ## Data and Classes
# - Create Dataloader class
# 
# Note: Working on Part (a) as of now.  
# Guiding light: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

# %%
START_TOKEN = "<START>"
END_TOKEN = "<END>"
UNK_TOKEN = "<UNK>"
PAD_TOKEN = "<PAD>"

class Vocabulary:
    def __init__(self, freq_dict, wd_to_id, id_to_wd):
        self.freq_dict = freq_dict
        self.wd_to_id = wd_to_id
        self.id_to_wd = id_to_wd
        self.N = len(freq_dict)
    
    def get_id(self, word):
        if word in self.wd_to_id:
            return self.wd_to_id[word]
        else:
            return self.wd_to_id[UNK_TOKEN]
        
    def decode(self, formula):
        """
        Input shape: (seq_len,)
        Output Shape: seq_len -> python list
        """
        decoded = []
        for wd in formula:
            if wd in self.wd_to_id:
                decoded.append(self.id_to_wd[wd])
            else:
                decoded.append(self.id_to_wd[UNK_TOKEN])

class LatexFormulaDataset(Dataset):
    """Latex Formula Dataset: Image and Text"""
    
    def __init__(self, csv_file, root_dir, transform = None, max_examples = None):
        """
        Arguments:
            csv_file (string): Path to the csv file with image name and text
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        info("Loading Dataset...")
        self.df = pd.read_csv(csv_file)
        info("Loaded.")
        #info("Loaded Dataset", self.df.info)
        
        #Slice the dataset if max_examples is not None
        if max_examples is not None:
            self.df = self.df.iloc[:max_examples, :]

        self.root_dir = root_dir
        self.transform = transform

        self.df['formula'] = self.df['formula'].apply(lambda x: x.split())
        self.df['formula'] = self.df['formula'].apply(lambda x: [START_TOKEN] + x + [END_TOKEN])

        self.maxlen = 0
        for formula in self.df['formula']:
            if len(formula) > self.maxlen:
                self.maxlen = len(formula)
        
        def convert_to_ids(formula):
            form2 = [self.vocab.get_id(wd) for wd in formula]
            return torch.tensor(form2, dtype=torch.int64)
        
        self.df['formula'] = self.df['formula'].apply(lambda x: x +[PAD_TOKEN]*(self.maxlen - len(x)))
        self.vocab= self.construct_vocab() 
        self.df['formula'] = self.df['formula'].apply(convert_to_ids)
        

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Returns sample of type image, textformula
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.df.iloc[idx, 0])
        image = io.imread(img_name)
        formula = self.df.iloc[idx, 1]

        # formula = np.array([formula], dtype=str).reshape(-1, 1)
        # formula = [self.vocab.get_id(wd[0]) for wd in formula]
        
        sample = {'image': image, 'formula': formula}

        if self.transform:
            sample['image'] = self.transform(sample['image'])
            
        return sample 
    
    def construct_vocab(self):
        """
        Constructs vocabulary from the dataset formulas
        """
        #Split on spaces to tokenize
        freq_dict = {}
        for formula in self.df['formula']:
            for wd in formula:
                if wd not in freq_dict:
                    freq_dict[wd] = 1
                else:
                    freq_dict[wd] += 1
        freq_dict[UNK_TOKEN] = 1
        N = len(freq_dict)
        wd_to_id = {}
        for i, wd in enumerate(freq_dict):
            wd_to_id[wd] = i
        id_to_wd = {v: k for k, v in wd_to_id.items()}
    
        #pad the formulas with 
        return Vocabulary(freq_dict, wd_to_id, id_to_wd)      

def get_dataloader(csv_path, image_root, batch_size, transform = None, max_examples = None, shuffle =True):
    """
    Returns dataloader,dataset for the dataset
    """
    dataset = LatexFormulaDataset(csv_path, image_root, max_examples=max_examples,transform=transform) #checked
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader, dataset
 

# %% [markdown]
# ## Encoder Network
# - A CNN to encode image to more meaningful vector

# %%
class EncoderCNN(nn.Module):
    def __init__(self):
        super().__init__()
    
        #@TODO:reduce number of layers: eliminate pools and acts
        self.conv1 = nn.Conv2d(3, 32, (5, 5))       
        self.conv2 = nn.Conv2d(32, 64, (5, 5))
        self.conv3 = nn.Conv2d(64, 128, (5, 5))        
        self.conv4 = nn.Conv2d(128, 256, (5, 5))        
        self.conv5 = nn.Conv2d(256, 512, (5, 5))
        
        self.pool = nn.MaxPool2d((2, 2))
        self.avg_pool = nn.AvgPool2d((3, 3))
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        
        x = F.relu(self.conv5(x))
        x = self.pool(x)
        
        x = self.avg_pool(x)
        x = x.view(-1,512) 
        # info(f"Encoder Output Shape: {x.shape}")
        return x
    
class DecoderLSTM(nn.Module):
    """
    Inputs:
    (here M is whatever the batch size is passed)

    context_size : size of the context vector [shape: (1,M,context_size)]
    n_layers: number of layers [for our purposes, defaults to 1]
    hidden_size : size of the hidden state vectors [shape: (n_layers,M,hidden_size)]
    embed_size : size of the embedding vectors [shape: (1,M,embed_size)]
    vocab_size : size of the vocabulary
    max_length : maximum length of the formula
    """
    def __init__(self, context_size, vocab, max_seq_len, n_layers = 1, hidden_size = 512, embed_size = 512):
        super().__init__()
        self.context_size = context_size
        self.vocab = vocab
        self.vocab_size = vocab.N
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.input_size = context_size + embed_size
        self.max_seq_len = max_seq_len

        self.embed = nn.Embedding(self.vocab_size, embed_size)
        self.lstm = nn.LSTMCell(self.input_size, self.hidden_size)
        self.linear = nn.Linear(hidden_size, self.vocab_size)
    
    def forward(self, context, target_tensor = None):
        """
        M: batch_size
        context is the context vector from the encoder [shape: (M,context_size)]
        target_tensor is the formula in tensor form [shape: (M,max_length)] (in the second dimension, it is sequence of indices of formula tokens)
            if target_tensor is not None, then we are in Teacher Forcing mode
            else normal jo bhi (last prediction ka embed is concatenated)
        """
        context.to(device)
        batch_size = context.shape[0]

        #initialize hidden state and cell state
        hidden = context
        cell = torch.zeros(batch_size, self.hidden_size).to(device)

        #initialize the input with embedding of the start token. Expand for batch size.
        init_embed = self.embed(torch.tensor([self.vocab.wd_to_id[START_TOKEN]]).repeat(batch_size, 1)).to(device)
        print("Embed Shape", init_embed.shape)
        #initialize the output_history and init_output
        outputs = []
        output = torch.zeros((batch_size, self.vocab_size)).to(device)
        
        
        for i in range(self.max_seq_len):
            #teacher forcing: 50% times
            r = torch.rand(1)
            if r>0.5 and target_tensor is not None:
                if i==0 :
                    embedding = init_embed
                else: 
                    embedding = self.embed(target_tensor[:, i-1]).reshape((batch_size, self.embed_size)).to(device)            
            else:
                if i==0 :
                    embedding = init_embed
                else:
                    #create embedding from previous input
                    embedding = self.embed(torch.argmax(output, dim = 1))

            lstm_input = torch.cat([context, embedding], dim = 1).to(device)
    
            hidden, cell = self.lstm(lstm_input, (hidden, cell))
            output = self.linear(hidden)
            outputs.append(output)
            
        output_tensor = torch.stack(outputs).permute(1,0,2) #LBV - > BLV

        return output_tensor, hidden, cell

# %% [markdown]
# ### Vocabulary
# - https://github.com/harvardnlp/im2markup/blob/master

# %% [markdown]
# ### Complete Network

# %%
class HandwritingToLatexModel(nn.Module):
    def __init__(self, context_size, vocab, max_seq_len, n_layers, hidden_size, embed_size):
        super().__init__()
        self.encoder = EncoderCNN()
        self.decoder = DecoderLSTM(context_size, vocab, max_seq_len, n_layers, hidden_size, embed_size)
    
    def forward(self, image, target_tensor = None):
        context = self.encoder(image)
        outputs, hidden, cell = self.decoder(context, target_tensor)
        return outputs

# %% [markdown]
# ### Utility Functions

# %%
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from tqdm import tqdm

plt.switch_backend('agg')
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))
    
def saveModel(save_path, model_state, optimiser_state, loss):
    torch.save({
            'model_state_dict': model_state,
            'optimizer_state_dict':optimiser_state,
            'loss': loss,  
    }, save_path)

# %% [markdown]
# ### Training Code.
# - Dataloader automatically loads in batches. The data need not be modified by us.

# %%
def generated_formula(output, vocab):
    """
    output: [shape: (Max_length,vocab_size)]
    """
    output = torch.argmax(output, dim = 1)
    output = output.tolist()
    formula = ' '.join([vocab.id_to_wd[id] for id in output])
    return formula

# %%
def train_epoch(dataloader,model, optimizer, criterion):
    total_loss = 0
    idx = 0
    pb = tqdm(dataloader, desc="Batch")
    for data in pb:
        idx+=1

        input_tensor, target_tensor = data['image'].to(device), data['formula'].to(device)
        outputs = model(input_tensor, target_tensor)
        train_dataset = dataloader.dataset 
        if(train_dataset and idx%500==0):
            generated_formula = [train_dataset.vocab.id_to_wd[token.item()] for token in torch.argmax(outputs, dim=2)[0]]
            required_formula = [train_dataset.vocab.id_to_wd[token.item()] for token in target_tensor[0]]
            print(f"Generated: {' '.join(generated_formula)}")
            print(f"Actual: {' '.join(required_formula)}")

        output_logits = outputs.permute(0,2,1)
        
        loss = criterion(
            output_logits,
            target_tensor
        )
        
        #backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 

        total_loss += loss.item()
        pb.set_description(f"Loss: {loss.item()}")

    return total_loss / len(dataloader)

def train(train_dataloader, model, n_epochs, optimizer = None, learning_rate=0.001, print_every=1, save_interval=2, save_prefix = 'model'):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every

    if not optimizer: optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=train_dataloader.dataset.vocab.wd_to_id[PAD_TOKEN]).to(device) #as stated in assignment

    model.train()
    for epoch in range(1, n_epochs + 1):
        info(f"Epoch {epoch}")
        loss = train_epoch(train_dataloader, model, optimizer, criterion)
        print_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
    
        if epoch % save_interval == 0:
            saveModel(f'/kaggle/working/{save_prefix}_epoch_{epoch}.pt', model.state_dict(), optimizer.state_dict(), loss)    
                
        info('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs), epoch, epoch / n_epochs * 100, print_loss_avg))

# %% [markdown]
# ## Training

# %%

batch_size = 32
vocab_size = 1000
CONTEXT_SIZE = 512
HIDDEN_SIZE = 512
EMBED_SIZE = 512
MAX_EXAMPLES = 1000
# image processing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
])


if __name__ == "__main__":
    train_csv_path = "data/SyntheticData/train.csv"
    image_root_path = "data/SyntheticData/images/"
    train_dataloader, train_dataset = get_dataloader(train_csv_path, image_root_path, batch_size, transform, max_examples=None)

    # %% [markdown]
    # ### Create Model

    # %%
    #create a network instance
    model = HandwritingToLatexModel(CONTEXT_SIZE, train_dataset.vocab, n_layers=1, hidden_size= HIDDEN_SIZE, embed_size=EMBED_SIZE, max_seq_len=train_dataset.maxlen).to(device)

    # %% [markdown]
    # ### Train

    # %%
    train(train_dataloader, model, 20, save_interval=2, save_prefix = 'model3.0')

    # %%
    def load_from_checkpoint(checkpoint_path):
        model_dicts = torch.load(checkpoint_path, map_location=device)
        
        model = HandwritingToLatexModel(CONTEXT_SIZE, train_dataset.vocab, n_layers=1, hidden_size= HIDDEN_SIZE, embed_size=EMBED_SIZE, max_seq_len=train_dataset.maxlen).to(device)
        model.load_state_dict(model_dicts['model_state_dict'])
        
        optims = torch.optim.Adam(model.parameters(), lr=0.001)
        optims.load_state_dict(model_dicts['optimizer_state_dict'])

        return model, optims

    model, optim = load_from_checkpoint("checkpoints/model2.0+12_epoch_6.pt")

    #resume training from checkpoints
    losses = train(train_dataloader, model, 20, optimizer = optim, save_interval=2, save_prefix = 'model2.0+18')

