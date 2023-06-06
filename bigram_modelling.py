import math
from collections import defaultdict
from scipy.stats import mannwhitneyu

class NgramLanguageModel:
	"""
	Class with code to train and run n-gram language models with add-k smoothing.
	"""

	def __init__(self): 
		self.unigram_counts = defaultdict(int)
		self.bigram_counts = defaultdict(int)
		
		self.k = 0.01
		
	def train(self, infile='samiam.train'):
		"""Trains the language models by calculating n-gram counts from the corpus
		at the path given in the variable `infile`. 

		These counts should be accumulated on the unigram_counts and bigram_counts
		objects. Note that these must be referenced with a prefix of 'self.', e.g.:
			self.unigram_counts

		You can assume the training corpus contains one sentence per line, which is
		already tokenized for you and thus can be tokenized with simply sentence.split().

		Remember that you have to add a special '<s>' token to the beginning
		and '</s>' token to the end of each sentence to correctly estimate the
		probabilities. 
		
		To run properly with the autograder, make keys for your bigram_counts dict
		by joining with an underscore (e.g, 'GREEN_EGGS').

		Parameters
		----------
		infile : str (defaults to 'brown_news.train')
			File path to the training corpus.

		Returns
		-------
		None (updates class attributes self.*_counts)
		"""
	# >>> YOUR ANSWER HERE
		with open(infile) as f:
			corpus = f.readlines()

		# creating list and adding special tokens
		for sentence in corpus:
			sentence_list = sentence.split()
			sentence_list.append('</s>')
			sentence_list.insert(0, '<s>')
			
			# populating bigram dictionary

			for i in range(len(sentence_list)):
				if i < len(sentence_list) - 1:
					key = sentence_list[i] + '_' + sentence_list[i + 1]
					if key not in self.bigram_counts:
						self.bigram_counts[key] = 1
					else:
						self.bigram_counts[key] += 1

		
	

	def predict_bigram(self, sentence):
		"""Calculates the log probability of the given sentence using a bigram LM.
		
		Analogous to predict_unigram, but uses a bigram model instead.

		Reminder to incorporate sentence-start and sentence-end tokens that match what
		you used in training.

		Parameters
		----------
		sentence : str 
			A sentence for which to calculate the probability.

		Returns
		-------
		float
			The log probability of the sentence.
		"""
		# >>> YOUR ANSWER HERE
		probab = 0.0
		sentence = sentence.split()
		sentence.append('</s>')
		sentence.insert(0, '<s>')

		curr_prob = 0.0
		for i in range(len(sentence)):
			if i < len(sentence) - 1:
				key = sentence[i] + '_' + sentence[i + 1]
				num = self.bigram_counts[key] + self.k
				# denom is the number of times the previous word appears +  k smoothing
				denom = self.unigram_counts[sentence[i]] + (self.k *  len(self.unigram_counts.keys()))
				curr_prob += math.log(num/denom)


				


		
		return curr_prob
		# >>> END YOUR ANSWER


	def test_perplexity(self, test_file):
		"""Calculate the perplexity of the trained LM on a test corpus.

		This seems complicated, but is actually quite simple. 

		First we need to calculate the total probability of the test corpus. 
		We can do this by summing the log probabiities of each sentence in the corpus.
		
		Then we need to normalize (e.g., divide) this summed log probability by the 
		total number of tokens in the test corpus. The one tricky bit here is we need
		to augment this count of the total number of tokens by one for each sentence,
		since we're including the sentence-end token in these probability estimates.

		Finally, to convert this result back to a perplexity, we need to multiply it
		by negative one, and exponentiate it - e.g., if we have the result of the above
		in a variable called 'val', we will return math.exp(val). 

		This log-space calculation of perplexity is not super directly explained in
		the main text of the chapter, but it is related to information theory math
		that is described in section 3.7.

		Another nice explanation of this, for your reference, is here:
		https://towardsdatascience.com/perplexity-in-language-models-87a196019a94

		Parameters
		-------
		test_file : str
			File path to a test corpus.
			(assumed pre-tokenized, whitespace-separated, one line per sentence)

		ngram_size : str
			A string ('unigram' or 'bigram') specifying which model to use. 
			Use this variable in an if/else statement.

		Returns
		-------
		float  
			The perplexity of the corpus (normalized total log probability).
		"""
		# >>> YOUR ANSWER HERE
		total_prob = 0.0
		total_token = 0
		# calculating total probability
		with open(test_file) as f:
			corpus = f.readlines()
		for sentence in corpus:
			total_prob += self.predict_bigram(sentence)
			total_token += len(sentence.split()) + 1

		
		# normalising step
		prob = total_prob / total_token
		prob = math.exp((prob * -1))

		
		return prob
	
# performing tests

	def stat_test(self, perp1, perp2):
		U, p_value = mannwhitneyu(perp1, perp2)

		return p_value


if __name__ == '__main__':
	ngram_lm = NgramLanguageModel()
	ngram_lm.train('preprocessed_commentaries.txt')
	print('Training male questions perplexity:\t', ngram_lm.test_perplexity('male_questions.txt'))
	print('Training female questions perplexity:\t', ngram_lm.test_perplexity('female_questions.txt'))
	print('Training male questions2 perplexity:\t', ngram_lm.test_perplexity('male_questions2.txt'))
	print('Training female questions perplexity:\t', ngram_lm.test_perplexity('female_questions2.txt'))



	# print('p_value for first', ngram_lm.stat_test(623.3249022829553,1143.8633877231598))
	# print('p_value for second', ngram_lm.stat_test(7005.479716400876,6766.818344610357))





