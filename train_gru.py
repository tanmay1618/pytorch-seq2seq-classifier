# coding: utf-8
from utils import load_data,load_word_emb,embeddedTensorFromSentence, normalizeString
import numpy as np
from model import EncoderRNN, AttnDecoderRNN, DecoderRNN
import torch
from torch import optim
import torch.nn as nn
from test_gru import evaluate, showAttention
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import pickle

device = torch.device("cpu")
N_word=100
B_word=6
hidden_size = 256
max_length = 1000
SOS_token = 0
CLASS_size = 6


word_emb = load_word_emb('../glove/glove.%dB.%dd.txt'%(B_word,N_word))
full_table, classes_, weight_tensor = load_data(device)
train_df, test_df = train_test_split(full_table, test_size=0.2, random_state=42)
print(train_df.department_new.value_counts())
print(test_df.department_new.value_counts())
CLASS_size = len(classes_)
class_index = range(CLASS_size)
class_dict = dict(zip(classes_, class_index))

import time
import math

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

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

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.show()

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=max_length):
	encoder_hidden = encoder.initHidden()
	input_length = len(input_tensor)
	encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
	for ei in range(input_length):
		encoder_output, encoder_hidden = encoder(input_tensor[ei],encoder_hidden)
		encoder_outputs[ei] += encoder_output[0, 0]
	decoder_hidden = encoder_hidden
	decoder_output, decoder_hidden = decoder(decoder_hidden)
	#decoder_output, decoder_hidden, decoder_attention = decoder(decoder_hidden, encoder_outputs)
	topv, topi = decoder_output.topk(1)
	decoder_input = topi.squeeze().detach()
	loss = criterion(decoder_output, torch.max(target_tensor, 1)[1])
	loss.backward()
	encoder_optimizer.step()
	decoder_optimizer.step()
	return loss.item()

def trainIters(encoder, decoder,data_df, n_iters, print_every=1000, plot_every=100, learning_rate=0.05):
	start = time.time()
	plot_losses = []
	print_loss_total = 0  # Reset every print_every
	plot_loss_total = 0  # Reset every plot_every
	encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
	decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
	criterion = nn.NLLLoss()#weight = weight_tensor)
	for iter in range(1, n_iters + 1):
		#print(iter)
		sentence = train_df.iloc[iter - 1]["description"]
		sentence = normalizeString(sentence)
		input_tensor = embeddedTensorFromSentence(sentence,device,word_emb,N_word)
		target_class = data_df.iloc[iter - 1]["department_new"]
		class_index = []
		for i in range(CLASS_size):
			class_index.append(0)
		class_index[class_dict[target_class]] = 1
		#import pdb; pdb.set_trace();
		#print(class_index)
		target_tensor = torch.tensor(class_index,dtype = torch.long ,device=device).view(1,CLASS_size)
		loss = train(input_tensor, target_tensor, encoder,
			     decoder, encoder_optimizer, decoder_optimizer, criterion)
		print_loss_total += loss
		plot_loss_total += loss
		if iter % print_every == 0:
			print_loss_avg = print_loss_total / print_every
			print_loss_total = 0
			print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
						 iter, iter / n_iters * 100, print_loss_avg))

		if iter % plot_every == 0:
			plot_loss_avg = plot_loss_total / plot_every
			plot_losses.append(plot_loss_avg)
			plot_loss_total = 0
	showPlot(plot_losses)

def evaluateTest(encoder,decoder):
	test_size = test_df.shape[0]
	y_true = []
	y_pred = []
	for iter in range(0, test_size + 1):
		sentence = test_df.iloc[iter - 1]["description"]
		sentence = normalizeString(sentence)
		input_tensor = embeddedTensorFromSentence(sentence,device,word_emb,N_word)
		target_class = test_df.iloc[iter - 1]["department_new"]
		class_index = []
		target_index = class_dict[target_class]
		#print(target_index)
		y_true.append(target_index)
		output = evaluate(encoder, decoder, input_tensor,max_length,device)
		topv, topi = output.topk(1)
		y_pred.append(topi.numpy()[0][0])
	cnf_matrix = confusion_matrix(y_true, y_pred)
	print("Accuarcy")
	print(accuracy_score(y_true, y_pred))
	print(cnf_matrix)
		
encoder = EncoderRNN(N_word, hidden_size).to(device)
encoder.apply(init_weights)
decoder = DecoderRNN(hidden_size, CLASS_size).to(device)
decoder.apply(init_weights)
n_iterations = train_df.shape[0]
trainIters(encoder, decoder, n_iterations, print_every=50, plot_every=10)
print(classes_)
evaluateTest(encoder,decoder)
