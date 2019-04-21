import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from loader import Loader
from tqdm import tqdm
from torch.utils.data import DataLoader

torch.manual_seed(1)

CONTEXT_SIZE = 2  # 2 words to the left, 2 to the right
EMBEDDING_DIM = 20
# raw_text = """We are about to study the idea of a computational process.
# Computational processes are abstract beings that inhabit computers.
# As they evolve, processes manipulate other abstract things called data.
# The evolution of a process is directed by a pattern of rules
# called a program. People create programs to direct processes. In effect,
# we conjure the spirits of the computer with our spells.""".split()

FILE='ALL'
loader=Loader()
raw_text=loader.get_chars_list()
# By deriving a set from `raw_text`, we deduplicate the array
vocab = set(raw_text)
vocab_size = len(vocab)

word_to_ix = {word: i for i, word in enumerate(vocab)}
data = []
for i in range(2, len(raw_text) - 2):
    context = [raw_text[i - 2], raw_text[i - 1],
               raw_text[i + 1], raw_text[i + 2]]
    target = raw_text[i]
    data.append((context, target))
print('data length is {}'.format(len(data)))


class CBOW(nn.Module):
    def __init__(self, vocab_size, context_size, embedding_dim):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * 2 * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        embeds=embeds.view(1, -1)
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        out = F.log_softmax(out, dim=1)
        return out


# create your model and train.  here are some functions to help you make
# the data ready for use by your module


def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    return torch.tensor(idxs, dtype=torch.long)
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
batch_size=2
make_context_vector(data[0][0], word_to_ix)  # example
losses = []
loss_function = nn.NLLLoss()
model = CBOW(vocab_size, CONTEXT_SIZE, EMBEDDING_DIM)
model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001)

data_iter = DataLoader(data,shuffle=False, num_workers=4)
for epoch in tqdm(range(100)):
    #print(epoch)
    total_loss = 0
    for context, target in data_iter:

        context=[a[0] for a in context]#if use Dataloader
        target=target[0]


        context_idxs = make_context_vector(context, word_to_ix)
        context_idxs.to(device)

        model.zero_grad()
        log_probs1 = model(context_idxs)
        loss = loss_function(log_probs1, torch.tensor([word_to_ix[target]], dtype=torch.long).to(device))
        total_loss += loss
        loss.backward()
        optimizer.step()
    losses.append(total_loss)

print(losses)
