from flask import Flask, request, make_response
from flask_cors import CORS, cross_origin
from flask_restful import Resource, Api
from flask import json
from json import dumps
from flask_jsonpify import jsonify
import os
import csv as csv
import nltk
import re
from collections import Counter
from nltk.corpus import wordnet
import sklearn
import string
import keras
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Model, load_model
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
#from PyDictionary import PyDictionary
#dictionary=PyDictionary()

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
app = Flask(__name__)
api = Api(app)

CORS(app)

category_cache = []
solution_cache = []
selected_index = []
index_selected_ele = []

@app.route("/")
def hello():
    return jsonify({'text':'Hello World!'})       

class InputQuery(Resource):
    def words(text): return re.findall(r'\w+', text.lower())

    WORDS = Counter(words(open('big_one1.txt').read()))
    
    def P(self,word, N=sum(WORDS.values())):
        "Probability of `word`."
        return self.WORDS[word] / N

    def correction(self,word): 
        "Most probable spelling correction for word."
        return max(self.candidates(word), key=self.P)

    def candidates(self,word): 
        "Generate possible spelling corrections for word."
        return (self.known([word]) or self.known(self.edits1(word)) or self.known(self.edits2(word)) or self.known(self.edits3(word)) or [word])

    def known(self,words): 
        "The subset of `words` that appear in the dictionary of WORDS."
        return set(w for w in words if w in self.WORDS)

    def edits1(self,word):
        "All edits that are one edit away from `word`."
        letters    = 'abcdefghijklmnopqrstuvwxyz'
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        deletes    = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
        inserts    = [L + c + R               for L, R in splits for c in letters]
        print("Edits1",set(deletes + transposes + replaces + inserts))
        return set(deletes + transposes + replaces + inserts)

    def edits2(self,word): 
        "All edits that are two edits away from `word`."
        return (e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))

    def edits3(self,word): 
        "All edits that are two edits away from `word`."
        return (e3 for e1 in self.edits1(word) for e2 in self.edits1(e1) for e3 in self.edits1(e1))
    
    def get(self,text):
        print("Inside Input Query")
        Query = text.lower()
        print('Query',Query.split(" "))
        f = open('count.csv')
        reader = csv.DictReader(f)
        for row in reader:
            Count=row["count"]
            K=row["k"]
            print("count...",Count)
            print("k...",K)
        #return(Count)
        #return(K)
        #if len(Query.split(" "))==1:
        
        Query_to_correct = Query.split(" ")
        temp=[]
        for i in range(len(Query_to_correct)):
            temp.append(self.correction(Query_to_correct[i]))
        Query_Corrected= " ".join(temp)
        #Query = self.correction(Query)

        print('Corrected Query', Query_Corrected)
        
        
        if(Query_Corrected == "Yes" or Query_Corrected== "No" or Query_Corrected == "yes" or Query_Corrected== "no"):
            if(Query_Corrected == "Yes" or Query_Corrected=="yes" and int(Count)==0):
                temp = solution_cache[0]
                print('count 0 and query yes')
                with open('count.csv', 'w') as csvfile:
                    csvfile.write("count," +"k,"+"\n")
                    csvfile.write("0,"+"0")
                del solution_cache[:]
                del category_cache[:]
                    #return(solution_input_quetry)
                return(temp)
            
            elif(Query_Corrected== "No" or Query_Corrected=="no" and int(Count)==0):
                print('count 0 and query no')
                with open('count.csv', 'w') as csvfile:
                    csvfile.write("count," +"k,"+"\n")
                    csvfile.write("1,"+"1")
                return("Is your query based on "+category_cache[1]+ " category?")
            
            elif(Query_Corrected == "Yes" or Query_Corrected=="yes" and int(Count)==1):
                temp = solution_cache[1]
                print('count 1 and query yes')
                with open('count.csv', 'w') as csvfile:
                    csvfile.write("count," +"k,"+"\n")
                    csvfile.write("0,"+"0")
                del solution_cache[:]
                del category_cache[:]
                return(temp)
                
                   #return(solution_input_query)
            elif(Query_Corrected== "No" or Query_Corrected=="no" and int(Count)==1):
                print('count 1 and query no')
                with open('count.csv', 'w') as csvfile:
                    csvfile.write("count," +"k,"+"\n")
                    csvfile.write("2,"+"1")
                return("Is your query based on "+category_cache[2]+ " category?")
            elif(Query_Corrected == "Yes" or Query_Corrected=="yes" and int(Count)==2):
                temp = solution_cache[2]
                print('count 2 and query yes')
                with open('count.csv', 'w') as csvfile:
                    csvfile.write("count," +"k,"+"\n")
                    csvfile.write("0,"+"0")
                del solution_cache[:]
                del category_cache[:]
                return(temp)

            elif(Query_Corrected== "No" or Query_Corrected=="no" and int(Count)==2):
                print('count 2 and query no')
                with open('count.csv', 'w') as csvfile:
                    csvfile.write("count," +"k,"+"\n")
                    csvfile.write("3,"+"1")
                return("Is your query based on "+category_cache[3]+ " category?")
            elif(Query_Corrected == "Yes" or Query_Corrected=="yes" and int(Count)==3):
                temp = solution_cache[3]
                print('count 2 and query yes')
                with open('count.csv', 'w') as csvfile:
                    csvfile.write("count," +"k,"+"\n")
                    csvfile.write("0,"+"0")
                del solution_cache[:]
                del category_cache[:]
                return(temp)
            elif(Query_Corrected== "No" or Query_Corrected=="no" and int(Count)==3):
                print('count 3 and query no')
                with open('count.csv', 'w') as csvfile:
                    csvfile.write("count," +"k,"+"\n")
                    csvfile.write("4,"+"1")
                return("Is your query based on "+category_cache[4]+ " category?")
            elif(Query_Corrected == "Yes" or Query_Corrected=="yes" and int(Count)==4):
                temp = solution_cache[4]
                print('count 4 and query yes')
                with open('count.csv', 'w') as csvfile:
                    csvfile.write("count," +"k,"+"\n")
                    csvfile.write("0,"+"0")
                del solution_cache[:]
                del category_cache[:]
                return(temp)
            else:
                with open('count.csv', 'w') as csvfile:
                    csvfile.write("count," +"k,"+"\n")
                    csvfile.write("0,"+"0")
                    
                readdata_time = csv.reader(open('Time.csv','r'))
                data_time = []
                for row in readdata_time:
                    data_time.append(row)
                Header = data_time[0]
                data_time.pop(0)
                r_time= pd.DataFrame(data_time, columns=Header)
                assignee = list(r_time.Assignee)
                print('Assignee', assignee)
                jobs = list(r_time.Jobs)
                print('Jobs',jobs)
                time = list(r_time.Time)
                print('time',time)
                category = list(r_time.Category)
                print('category',category)
                assigne_name=""
                total_time = 0
                jobs_new = 0
                time_new = 0
                for i in range(len(category)):
                    print('Category of i',category[i], category_cache[0])
                    print('Jobs of i',jobs[i])
                    if category[i] == category_cache[0]:
                        print('selected category',category[i])
                            #a = i
                        assigne_name = assignee[i]
                        jobs_new = int(jobs[i])+1
                        time_new = int(time[i])
                    total_time = jobs_new * time_new
                    print("total time", total_time)
                    #with open('Time.csv', 'w') as csvfile:
                     #   csvfile.write("count," +"k,"+"\n")
                      #  csvfile.write("0,"+"0")  
                del solution_cache[:]
                del category_cache[:]
                    #return("we are escalated your problem to an Engineer",total_time)
                return("we are escalated your problem to an Engineer "+assigne_name+" ,he will response to your answer within "+str(total_time)+" days")
        elif 'esa version' in Query_Corrected:
            stop_words = set(stopwords.words('english'))
            #text1 = input("Enter the query:")
            word_tokens=word_tokenize(Query_Corrected)
            filtered_sentence = [w for w in word_tokens if not w in stop_words]
            new_text1 = [" ".join(filtered_sentence)]
            Query_Corrected=[b for l in new_text1 for b in zip(l.split(" ")[:-1],l.split(" ")[1:])]
            print(Query_Corrected)
            bigram_data=[]
            value=[]
            value1=[]
            version = []
            p1=''
            p2=''
            p3=''
            p4=''
            for i in range(len(Query_Corrected)):
                if Query_Corrected[i] == ('esa','version'):
                    for j in range(len(Query_Corrected[i])):
                        bigram_data.append(Query_Corrected[i][j])
                    version = " ".join(new_text1)
                    version = word_tokenize(version)
                    for word in bigram_data:
                        if word in version:
                            version.remove(word)
                            print(version)
                    for word1 in version:
                        if word1=='latest':
                            p1='latest'
                            for x in version:
                                if word1==x:
                                    version.remove(word1)
                                    break
                        elif word1=='older':
                            p1='older'
                            for x in version:
                                if word1==x:
                                    version.remove(word1)
                                    break
                
                            
                else:
                    p1=''
                    for version_number in version:
                        if version_number == '4':
                            p2='4'
                            for x in version:
                                if version_number == x:
                                    version.remove(version_number)
                                    break
                        elif version_number == '16':
                            p2='16'
                            for x in version:
                                if version_number == x:
                                    version.remove(version_number)
                                    break
                        elif version_number == '17':
                            p2='17'
                            for x in version:
                                if version_number == x:
                                    version.remove(version_number)
                                    break
                        elif version_number == '18':
                            p2='18'
                            for x in version:
                                if version_number == x:
                                    version.remove(version_number)
                                    break
                        else:
                            p2=''
                    for data in version:
                        if data == 'cp1':
                            p3='cp1'
                            for x in version:
                                if data == x:
                                    version.remove(data)
                                    break
                        elif  data == 'cp2':
                            p3='cp2'
                            for x in version:
                                if data == x:
                                    version.remove(data)
                                    break
                        elif  data == 'cp3':
                            p3='cp3'
                            for x in version:
                                if data == x:
                                    version.remove(data)
                                    break
                        elif  data == 'cp4':
                            p3='cp4'
                            for x in version:
                                if data == x:
                                    version.remove(data)
                                    break
                        elif data == 'cp5':
                            p3='cp5'
                            for x in version:
                                if data == x:
                                    version.remove(data)
                                    break
                        elif  data == 'ep1':
                            p3='ep1'
                            for x in version:
                                if data == x:
                                    version.remove(data)
                                    break
                        elif  data == 'ep2':
                            p3='ep2'
                            for x in version:
                                if data == x:
                                    version.remove(data)
                                    break
                        else:
                            p3=''
                    for idata in version:
                        if idata == 'icp1':
                            p4='icp1'
                            for x in version:
                                if idata == x:
                                    version.remove(idata)
                                    break
                        elif idata == 'icp2':
                            p4='icp2'
                            for x in version:
                                if idata == x:
                                    version.remove(idata)
                                    break
                        else:
                            p4=''
                if(p1=='latest' and p2=='' and p3=='' and p4==''):
                    return("latest version of esa is 18.0")
                elif(p1=='older' and p2=='' and p3=='' and p4==''):
                    return("older version of esa is 4.0")
                elif(p2=='4' and p3=='cp1' and p4=='icp1'):
                    return("Build numbers: 4.0.1.1217 and Release date: 19-06-12" )
                    
                elif(p2=='4' and p3=='cp1' and p4=='icp2'):
                    print('4 and cp1 and icp2')
                elif(p2=='4' and p3=='cp2' and p4=='icp1'):
                    return("Build numbers: 4.0.2.1613 and Release date: 26-04-13" )
                    
                elif(p2=='4' and p3=='cp2' and p4=='icp2'):
                    
                    print("4 and cp2 and icp2")
                elif(p2=='4' and p3=='cp3' and p4=='icp1'):
                    return("Build numbers: 4.0.3.1854 and Release date: 25-02-14" )
                    
                elif(p2=='4' and p3=='cp3' and p4=='icp2'):
                    print("4 and cp3 and icp2")
                elif(p2=='4' and p3=='cp4' and p4=='icp1'):
                    print("4 and cp4 and icp1")
                elif(p2=='4' and p3=='cp4' and p4=='icp2'):
                    print("4 and cp4 and icp2")
                elif(p2=='4' and p3=='cp5' and p4=='icp1'):
                    return("Build numbers: 4.0.5.2274 and Release date: 04-09-15" )
                    
                elif(p2=='4' and p3=='cp5' and p4=='icp2'):
                    return("Build numbers: 4.0.5.2375 and Release date: 17-06-16" )
                    
                elif(p2=='4' and p3=='cp1' and p4==''):
                    return("Build numbers: 4.0.1.1166 and Release date: 28-03-12" )
                    #print("Build numbers: 4.0.1.1166")
                    #print("Release date: 28-03-12")
                    #print("**")
                elif(p2=='4' and p3=='cp2' and p4==''):
                    return("Build numbers: 4.0.2.1445 and Release date: 23-11-12" )
                    
                elif(p2=='4' and p3=='cp3' and p4==''):
                    return("Build numbers: 4.0.3.1757 and Release date: 20-09-13" )
                    
                elif(p2=='4' and p3=='cp4' and p4==''):
                    return("Build numbers: 4.0.4.1993 and Release date: 29-09-14" )
                    
                elif(p2=='4' and p3=='cp5' and p4==''):
                    return("Build numbers: 4.0.5.2230 and Release date: 27-02-15" )
                    
                elif(p2=='4' and p3=='' and p4==''):
                    return("Components are - Build number: 4.0.0.986 and Release date: 30-09-11, CP1,CP2,CP3,CP4,CP5" )
                    
                elif(p2=='16' and p3=='ep1' and p4==''):
                    return("Build numbers: 16.0.0.654 and Release date: 20-05-16" )
                    
                elif(p2=='16' and p3=='ep2' and p4==''):
                    return("Build numbers: 16.0.0.667 and Release date: 26-05-16" )
                    
                elif(p2=='16' and p3=='' and p4==''):
                    return("Build numbers: 16.0.0.574 and Release date: 04-03-16" )
                    
                elif(p2=='17' and p3=='' and p4==''):
                    return("Build numbers: 17.0.0.100 and Release date: 29-05-17" )
                    
                elif(p2=='18' and p3=='' and p4==''):
                    return("Build numbers: 17.0.0.200 and Release date: 29-03-18" )
        elif(Query_Corrected in selected_index):
            readdata = csv.reader(open('opm_finder_updated_new.csv','r'))
            data = []
            for row in readdata:
                data.append(row)
            Header = data[0]
            data.pop(0)
            r= pd.DataFrame(data, columns=Header)
            Solution = list(r['Probable Solution'].values)
            for i in range(len(index_selected_ele)):
                if(i+1 == int(Query_Corrected)):
                    return(Solution[index_selected_ele[i]])
        
        else:
            readdata = csv.reader(open('opm_finder_updated_new.csv','r'))
            data = []
            for row in readdata:
                data.append(row)
            Header = data[0]
            data.pop(0)
            r= pd.DataFrame(data, columns=Header)
            Text = list(r.Description)
            print("Initial Text",Text)
            Key = list(r.Keywords)
            print("Initial Key",Key)
            Categories = list(r.Categories)
            print("Initial Categories",Categories)
            Assignee = list(r.Assignee)
            print("Initial Assignee",Assignee)
            Time = list(r['Time Taken'].values)
            Solution = list(r['Probable Solution'].values)
            print('strip....................................',Text)
            Text_s = [s.strip() for s in Text]
            print(Text)

            encoder = LabelBinarizer()
            encoder.fit(Categories)
            Text_labels = encoder.transform(Categories)

            readdata_greetings = csv.reader(open('opm_finder_updated_new_sample.csv','r'))
            data_greetings = []
            for row in readdata_greetings:
                data_greetings.append(row)
            Header_greetings = data_greetings[0]
            data_greetings.pop(0)
            r_greetings= pd.DataFrame(data_greetings, columns=Header_greetings)
            Text_greetings = list(r_greetings.Description)
            print("Initial Text _greetings",Text_greetings)
            Solution_greetings = list(r_greetings['Probable Solution'].values)
            solution_of_greetings = ""
            bool_check = False
            for i in range(len(Text_greetings)):
                if Query_Corrected == Text_greetings[i]:
                    print(Text_greetings[i],i)
                    solution_of_greetings = Solution_greetings[i]
                    bool_check = True
                    print(bool_check)

            if bool_check == True:
                #return the solution
                print("solution",solution_of_greetings)
                return(solution_of_greetings)
            else:
                fullQuery=Query_Corrected.split(" ")

                print('full1',fullQuery)

                #if(Query.)
                stopWordsQuery = set(stopwords.words('english'))
                operators = set(('and', 'but', 'not'))
                stopWordsQuery=set(stopwords.words('english')) - operators 
                
                fullnewQuery= []
                tmpQuery=[] 

                for d in range(len(fullQuery)):
                    words = word_tokenize(fullQuery[d])
                    for w in words:
                        if w not in stopWordsQuery:
                            tmpQuery.append(w)
                    x=" ".join(tmpQuery).lower()
                    tmpQuery=[]
                    fullnewQuery.append(x)
                full_string_query = " ".join(fullnewQuery)
                tokenised_words_query = word_tokenize(full_string_query)

                print('tokenised word', tokenised_words_query)

                count = Counter(tokenised_words_query)

                for word in count:
                    count[word] /= float(len(count))

                print(count)
                query_word=[]
                frequency_input =0
                for word, freq in count.items():
                    frequency_input = freq
                    query_word.append(word)

                keywords_query=['akka','jgroup','esa']
                #boolean_check_value=[True for x in keywords_query if x in query_word]
                boolean_check_value=np.any(np.in1d(query_word, keywords_query))
                print('boolean_check_value..',boolean_check_value)
                if(frequency_input >= 0.3 and boolean_check_value==True) :
                    readdata_Query = csv.reader(open('ans_book.csv','r'))
                    data_Query = []
                    for row in readdata_Query:
                        data_Query.append(row)
                    Header_Query = data_Query[0]
                    data_Query.pop(0)
                    r_Query= pd.DataFrame(data_Query, columns=Header_Query)
                    common_word = list(r_Query.keywords)
                    question = list(r_Query.questions)
                    answer=list(r_Query.answers)

                    Question = ""
                    for i in range(len(common_word)):
                        for word in tokenised_words_query:
                            if(word == common_word[i]):
                                print(common_word[i])
                                print(question[i])
                                Question = question[i]
                                answer=answer[i]
                                return answer
                    #Here write code to add this question to text input query
                    #Text = Text + [Question]
                    #return(Question)         
                else:
                    Text = Text + [Query_Corrected]

                
                #operators = set('and', 'or', 'not')
                operators = set(('and', 'but', 'not'))
                stopWords=set(stopwords.words('english')) - operators 
                
                #stopWords = stopWords.remove('not')
                #print("without not...........................",stopWords)
                #specificStopWords = list('not')
                
                #stopWords = stopWords - specificStopWords
                #stopWords = specificStopWords - stopWords
            
                #specificStopWords1 = set('why')
                #stopWords = stopwords - specificStopWords1
                full1=[" ".join(data.split(" ")) for data in Text]
                
                print("Not..........................",stopWords)
            
                full= []
                tmp=[] 
                for d in range(len(full1)):
                    words = word_tokenize(full1[d])
                    for w in words:
                        if w not in stopWords:
                            tmp.append(w)
                    x=" ".join(tmp).lower()
                    tmp=[]
                    full.append(x)
                

                splited = []
                print('Full sentence',full)
                full = [''.join(c for c in s if c not in string.punctuation) for s in full]
                print("full text punctuation remove..........",full)
                for i in range(len(full)):
                    if 'but' in full[i]:
                        splited = re.split('but',full[i])
                        full[i] = ''
                        full[i] = splited[1].strip()
                        print('Splited Word',splited[1].strip())
                        full=re.sub(' +', ' ',full[i])
                        
                print('Full sentence after split',full)
                
                
                
                
                full_string = " ".join(full)
                tokenised_words = word_tokenize(full_string)

                unique_word=[]
                seen=set()
                for word in tokenised_words:
                    if word not in seen:
                        seen.add(word)
                        unique_word.append(word)
                print('Tokenized words',tokenised_words)
                print('Length Tokenized words',len(tokenised_words))
                print('Unique Tokenized words',unique_word)
                print('Length Unique Tokenized words',len(unique_word))
                tmpQuery=[]
                fullQuery=[]
                words = word_tokenize(Query_Corrected)
                for w in words:
                    if w not in stopWords:
                        tmpQuery.append(w)
                x=" ".join(tmpQuery)
                fullQuery.append(x)

                print('Processed',full)
                print('Processed query',fullQuery)
                str1 = ''.join(unique_word)
                synonyms = []

                for i in range(len(unique_word)):
                    for syn in wordnet.synsets(unique_word[i]):
                        for l in syn.lemmas():
                            synonyms.append(l.name())
                print("synonyms............",synonyms)

                print("synonyms are.............",set(synonyms))

                pos_words = []
                for i in range(len(full)):
                    tmp = []
                    words = word_tokenize(full[i])
                    for j in range(len(words)):
                        if 'not' in full[i]:
                            j = j + 1
                        tmp.append(j+1)
                    pos_words.append(tmp)
                print('Position of words',pos_words)

                matrix_words = []
                for i in range(len(full)):
                    sentence_tokenized_words = word_tokenize(full[i])
                    print("Sentence_tokenize_word",sentence_tokenized_words)
                    print("Length of sentence_tokenized_words", len(sentence_tokenized_words))
                    table= [ [ 0 for m in range(len(unique_word))]  for n in range(len(sentence_tokenized_words)) ]
                    table= np.array(table)
                    print("table in array.............",np.array(table))
                    for j in range(len(unique_word)):
                        for k in range(len(sentence_tokenized_words)):
                            if sentence_tokenized_words[k] == unique_word[j]:
                                print("Sentence tokenized word",sentence_tokenized_words[k])
                                print("Unique word",unique_word[j])
                                print("Position of matched words",k,j)
                                table[k][j]=1

                    matrix_words.append(table)
                print("Matrix words",matrix_words)
                print("Lenght of matrix", len(matrix_words))
                matrix_words=np.array(matrix_words)
                print("matrix_words_array.............",matrix_words)

                randomvalues = []
                r_values_no = np.random.randint(2, size=10)
                print("Random values unique word lenth",len(unique_word))
                #r_values = np.random.randn(10)
                for i in range(len(unique_word)):
                    np.random.seed(i)
                    randomvalues.append(np.random.randn(10))
                    #r_values = np.random.randn(10)
                    #randomvalues.append(r_values)
                    #r_values = r_values + 10
                print('len random values',len(randomvalues))
                print('random values',randomvalues)

                all_values = []
                for i in range(len(matrix_words)):
                    vec_sample = np.array([0,0,0,0,0,0,0,0,0,0])
                    for j in range(len(matrix_words[i])):
                        for k in range(len(matrix_words[i][j])):
                            if matrix_words[i][j][k] == 1:
                                print('positions',i,j,k)
                                print(' vec sample description ',vec_sample +  randomvalues[k] * pos_words[i][j])
                                vec_sample = (vec_sample +  randomvalues[k] * pos_words[i][j])
                    all_values.append(vec_sample)
        
                print('All values',all_values)
                print('Length of all values', len(all_values))

                n_dd = []
                for i in range(len(all_values)):
                    n_d=np.array([0,0,0,0,0,0,0,0,0,0],dtype='float32')
                    for j in range(len(all_values[i])):
                    #n_d=np.array([0,0,0,0,0,0,0,0,0,0])
                        n_d1=(all_values[i][j]-(min(all_values[i])))/((max(all_values[i]))-(min(all_values[i])))
                        print("n_d1...................",n_d1)
                        n_d[j]=n_d1
                        print("n_d..........................",n_d[j])
                    n_dd.append(n_d)
                print("all_normalize_data.................",n_dd)


                n_dd_new = list(n_dd[-1])
                n_dd_new = np.asarray(n_dd_new)
                n_dd_new = n_dd_new.reshape(1,10)
                print("last all values",n_dd_new)
            
                x_input = np.array(n_dd)
                print("x_input_shape..........",x_input.shape)
                print("shape of 0th x_input..........",x_input.shape[0])
                print("x_input..................",x_input)
                y_input = np.array(Text_labels)
                x_input = x_input.reshape(x_input.shape[0],x_input.shape[1],1)
                print("x_input_reshape..................",x_input)
     
                description_all_values = []
                solution_values = []
                for i in range(len(Solution)):
                    description_all_values.append(n_dd[i])
                    solution_values.append(Solution[i])
                print('All solution is',solution_values)
                print('All description values of category index',description_all_values)
                print('Length of all description',len(description_all_values))


    ############## for category ###################

                description_all_category_values = []
                category_values = []
                for i in range(len(Categories)):
                    description_all_category_values.append(n_dd[i])
                    category_values.append(Categories[i])
                print('All Category is',category_values)
                print('All description values of category index',description_all_category_values)
                print('Length of all description',len(description_all_category_values))

    ############# for greetings category #################

    ###### E distance for category ##############3

                e_distance = euclidean_distances(description_all_category_values,n_dd_new)
                print('E distance',e_distance)
                flattened_list = []
                to_be_returned_final=[]

                #flatten the list
                for x in e_distance:
                    for y in x:
                        flattened_list.append(y)
                print('Flattend list',flattened_list)
                final=[]
                final_pos=[]
                
                for i in range(len(flattened_list)):
                    if flattened_list[i] <= 0.0:
                        final.append(flattened_list[i])
                        final_pos.append(i)
                        print("Final",final)
                        print("Final Pos",final_pos)
                
                    if final_pos:
                        for i in final_pos:
                            to_be_returned_final.append(Solution[i])
                            str1 = ''.join(to_be_returned_final)
                        return(str1)
            
                threshold = 0.75
                flatten_selected = []
                for flatten_value in flattened_list:
                    if flatten_value < threshold:
                        flatten_selected.append(flatten_value)

                len_of_flatten_selected = len(flatten_selected)
                print('Length of ele selected', len_of_flatten_selected)
                print('Flatten Selected', flatten_selected)
                
                ele = np.argsort(flattened_list)
                ele_selected = ele[:len_of_flatten_selected]
                
                print('Element', ele)
                print('Element Selected', ele_selected)
                
                new_list = []
                while flattened_list:
                    minimum = flattened_list[0]  # arbitrary number in list 
                    for x in range(len(flattened_list)):
                        if flattened_list[x] < minimum:
                            minimum = flattened_list[x]
                    new_list.append(minimum)
                    flattened_list.remove(minimum)
                #print("sorted e distances......",sorted(e_distance)[:3])
                #new_distances=sorted(e_distance)[:3]
                print("new distances for category...",new_list)
                
                edist_arr = []
                output_response = csv.reader(open('output_response.csv','r'))
                output_response_array = []
                for row in output_response:
                    output_response_array.append(row)
                Header_output = output_response_array[0]
                output_response_array.pop(0)
                r_output= pd.DataFrame(output_response_array, columns=Header_output)
                r_output_count = list(r_output.k)
                for i in ele[:int(r_output_count[0])]:
                    edist_arr.append(i)
                print("e distance array_for_category",edist_arr)
                print('Element', ele)
                solution_input_query = ""
                category_input_query = ""
                if len(ele_selected)!=0:
                    to_be_returned_list = []
                    index_selected_ele.clear()
                    print('Element Selected before appending', ele_selected)
                    for i in ele_selected:
                        to_be_returned_list.append(Text[i])
                        index_selected_ele.append(i)
                    print("Flatten Description",to_be_returned_list)
                    selected_index.clear()
                    json_format =[]
                    for i in range(len(to_be_returned_list)):
                        selected_index.append(str(i+1))
                        json_format.append({i+1:to_be_returned_list[i]})
                    #json_format_data = json.dumps(to_be_returned_list)
                    print("Data in Json Format",json_format)
                    print("Selected Index",selected_index)
                    print("Element Selected Index",index_selected_ele)
                    data = {"results":json_format}
                    return jsonify(data)
                else:
                    for m in edist_arr:
                        category_cache.append(category_values[m])
                        solution_cache.append(solution_values[m])
                        print("Category cache",category_cache)
                        print("Solution cache",solution_cache)
                
                    if(int(Count)==0 and int(K)==0):
                        print('count 0 and k 0')
                        with open('count.csv', 'w') as csvfile:
                            csvfile.write("count," +"k"+"\n")
                            print("hi.....")
                            csvfile.write("0,"+"1")
                        return("Is your query based on "+category_cache[0]+ " category?")
               
    
api.add_resource(InputQuery, '/inputquery/<text>') # Route_4
   

if __name__ == '__main__':
     app.run(port=5002)
