import nltk
nltk.download('popular')
sentence = "hello, world"
tokens = nltk.word_tokenize(sentence)
print(tokens)