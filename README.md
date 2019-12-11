# Overview

## Preprocessing functions
* Clear string
* Replcae parts of sentences for specific purpose:
	* Slang and Abbreviations (to be done)
	* Replace Contractions
	* Remove Numbers
	* Replace Repetitions of Punctuation
	* Replace Negations with Antonyms
	* Remove Punctuation
	* Handling Capitalized Words
	* Lowercase
	* Replace Elongated Words
* Remove Stopwords 
* Lemmatizing 
* Stemming
* Padding

Above functionalities are available via a single pipeline function quickAnalyzer.preprocessPipeline

## Analysis Tool
* Bag of word
* LDA topic model
* General sentiment analysis
* PoS Tagging
* WordCloud

## Wrapper for embeddings
* Unified interface for embedding:
	* Flair embedding (chararcter level)
	* Sentence embedding
	* Word embedding
* Wrapper for Gensim embedding
	* Allow updating
	* Simple interface for out-of-voc word



