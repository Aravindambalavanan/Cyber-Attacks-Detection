import pickle
import re
import pandas as pd
import numpy as np

import os
import requests


_CWD = os.getcwd() 

final_RFC_FE_model_file = os.path.join(_CWD,'model','final_RFC_FE_model.model')	
tfidf_vec_file = os.path.join(_CWD,'model','tfidf_vec.sav')											
with open(final_RFC_FE_model_file, 'rb') as file:
	RFC_FE_model = pickle.load(file)
with open(tfidf_vec_file, 'rb') as file:
	TFIDF_vectorizer = pickle.load(file)

def clean_query(input_string):
	"""
	This function cleans the query by removing special characters and spaces.
	It also converts the query to lower case.
	input_string: string
	return: string
	"""
	cleaned = re.sub('[^a-zA-Z0-9\s]',' ',input_string)	
	cleaned = re.sub('\s{2,}',' ',cleaned)							
	return cleaned.lower().strip()											
def predict_probab(X_query):
	"""
	This function predicts the probability of the query being a SQLi attack.
	input: string
	Output: numpy array
	"""
	cleaned_query = clean_query(X_query)
	no_of_special_chars = len(re.findall('[^a-zA-Z0-9\s]',X_query)) 
	query_length = len(X_query.split())															
	no_of_num_eq_num = len(re.findall(r'\d\s*=\s*\d',X_query))			
	X_q = TFIDF_vectorizer.transform([cleaned_query])								
	X_q_tfidf_FE = pd.DataFrame(data=X_q.toarray(),columns=TFIDF_vectorizer.get_feature_names_out()) 
	X_q_tfidf_FE['num_sc'] = no_of_special_chars										
	X_q_tfidf_FE['q_len'] = query_length													
	X_q_tfidf_FE['num_eq_num'] = no_of_num_eq_num									
	proba = RFC_FE_model.predict_proba(X_q_tfidf_FE)							
	return proba[0]																									

def predict_class(X_q_obj, prob=False):
	"""
	This function predicts the class of the query if it is malicious or not.
	if prob is set to True, it returns the probability of the query being a SQLi attack.
	input: X_q_obj: string, prob: boolean
	output: boolean or numpy array
	"""
	if isinstance(X_q_obj,list):				
		proba = []
		for x in X_q_obj:
			proba.append(predict_probab(x))
		return np.array(proba)						
		
	if isinstance(X_q_obj, str):			
		proba = predict_probab(X_q_obj)		
		class_ = np.argmax(proba)					
		if prob:													
			return proba										
		else:															
			return class_										