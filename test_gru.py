
from utils import load_data,load_word_emb,embeddedTensorFromSentence
import torch
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')
import matplotlib.ticker as ticker

def evaluate(encoder, decoder, input_tensor, max_length, device):
	with torch.no_grad():
		input_length = len(input_tensor)
		encoder_hidden = encoder.initHidden()

		encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
		for ei in range(input_length):
			encoder_output, encoder_hidden = encoder(input_tensor[ei],encoder_hidden)
			encoder_outputs[ei] += encoder_output[0, 0]
		decoder_hidden = encoder_hidden
		decoder_output, decoder_hidden = decoder(decoder_hidden)
		topv, topi = decoder_output.topk(1)

	return decoder_output


def showAttention(input_sentence, attentions):
    words = input_sentence.split(' ')
    plt.bar(words,attentions[:,0:len(words)].numpy().tolist()[0],color='green')

    plt.show()

