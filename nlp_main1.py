import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

if __name__ == '__main__':
	
	stopwords = ['porfavor','dame','el','cual','es','cual','es']
	

	sentences = ["porfavor pon juanes","dame el clima de hoy  cual es el clima de hoy ","cual es tu nombre"]

	temp = []
	for sentence in sentences:

		tokens = sentence.split(" ")
		clean_tokens = []

		for token in tokens:
			if all(char in set(string.punctuation) for char in token):
				continue

			if token.isdigit():
				continue

			token = token.lower()
			token = token.strip()

			if token in stopwords:
				continue
		
			clean_tokens.append(token)

		temp.append(' '.join(clean_tokens))


	#bag of words transformation
	count_vect = CountVectorizer()
	bag_of_words_array = count_vect.fit_transform(temp)

	#model training
	naive_bayes_classifier = MultinomialNB()
	naive_bayes_classifier.fit(bag_of_words_array,['juanes','clima','nombre'])

	
	hola= "cual es el clima"
	hola2 = []

	tokens = hola.split(" ")

	#sentecne cleaning
	hola1 = []

	for token in tokens:
		if all(char in set(string.punctuation) for char in token):
			continue

		if token.isdigit():
			continue

		token = token.lower()
		token = token.strip()

		if token in stopwords:
			continue
	
		hola1.append(token)



	# only for this practice

	hola2.append(' '.join(hola1))
	

	#bag of words transformation
	count_vect2 = CountVectorizer()
	bag_of_words_array2 = count_vect.transform(hola2)




	#model predict
	print("predict...")
	print(naive_bayes_classifier.predict(bag_of_words_array2))

	"""
	Implementar 3 oraciones 

	"""