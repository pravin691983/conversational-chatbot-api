from flask import Flask
from flask_restful import Resource, Api, reqparse, abort, marshal, fields
import os

import nltk
from keras.models import load_model
import json
import random
import tensorflow as tf
from tensorflow import keras
import numpy as np
from pathlib import Path
from nltk.stem import WordNetLemmatizer
import json
import pickle

# Initialize Flask app
app = Flask(__name__)
port = int(os.environ.get("PORT", 5000))

# Initialise api
api = Api(app)

# Define list of books
# A List of Dicts to store all of the books
books = [{
    "id": 1,
    "title": "Zero to One",
    "author": "Peter Thiel",
    "length": 195,
    "rating": 4.17
},
    {
    "id": 2,
    "title": "Atomic Habits ",
    "author": "James Clear",
    "length": 319,
    "rating": 4.35
}
]

# load the model architecture

intents = json.loads(open('./intents.json').read())
words = pickle.load(open('./words.pkl', 'rb'))
classes = pickle.load(open('./classes.pkl', 'rb'))

# Download wordnet & punkt
nltk.download('punkt')
nltk.download('wordnet')

# Initilise Data
lemmatizer = WordNetLemmatizer()


def loadRetrivalModal():
    with open('./RetrievalBased_ChatBot_model.json', 'r') as json_file:
        json_savedModel = json_file.read()

    # load the model architecture
    model_retrieval_based = tf.keras.models.model_from_json(json_savedModel)

    my_file = Path('./RetrievalBased_ChatBot_weights.hdf5')

    if my_file.is_file():
        # file exists
        model_retrieval_based.load_weights(
            './RetrievalBased_ChatBot_weights.hdf5')

    return model_retrieval_based


def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(
        word.lower()) for word in sentence_words]
    # print(sentence_words)
    return sentence_words


def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return(np.array(bag))


def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def getResponse(ints, intents_json):
    list_of_intents = intents_json['intents']

    if ints:
        tag = ints[0]['intent']
        # print('List of intents :', list_of_intents)
        for i in list_of_intents:
            if(i['tag'] == tag):
                result = random.choice(i['responses'])
                break
    else:
        for i in list_of_intents:
            if(i['tag'] == 'noanswer'):
                result = random.choice(i['responses'])
                break

    # print('getResponse result :', result)
    return result


def chatbot_response(msg, model_retrieval_based):
    ints = predict_class(msg, model_retrieval_based)
    # print('predict_class response : ',ints)
    res = getResponse(ints, intents)
    return res


# Load model
model_retrieval_based = loadRetrivalModal()

# Schema For the Book Request JSON
chatMessageField = {
    "question": fields.String,
    "answer": fields.String
}

# Resource: Individual Book Routes


class ChatMessage(Resource):
    def __init__(self):
        # Initialize The Flsak Request Parser and add arguments as in an expected request
        self.reqparse = reqparse.RequestParser()

        self.reqparse.add_argument("question", type=str, location="json")
        self.reqparse.add_argument("answer", type=str, location="json")

    def post(self):
        args = self.reqparse.parse_args()
        print(args)
        reply = {
            "question": args["question"],
            "answer": chatbot_response(args["question"], model_retrieval_based)
        }
        print("Reply to sent .... !!!!")
        print(reply)
        # return {'Message': reply}

        return{"Predicted Response": marshal(reply, chatMessageField)}
        # return {"Reply": marshal(chatMessage, reply)}, 201


# Schema For the Book Request JSON
bookFields = {
    "id": fields.Integer,
    "title": fields.String,
    "author": fields.String,
    "length": fields.Integer,
    "rating": fields.Float
}

# Resource: Individual Book Routes


class Book(Resource):
    def __init__(self):
        # Initialize The Flsak Request Parser and add arguments as in an expected request
        self.reqparse = reqparse.RequestParser()
        self.reqparse.add_argument("title", type=str, location="json")
        self.reqparse.add_argument("author", type=str, location="json")
        self.reqparse.add_argument("length", type=int, location="json")
        self.reqparse.add_argument("rating", type=float, location="json")

        super(Book, self).__init__()

    # GET - Returns a single book object given a matching id
    def get(self, id):
        book = [book for book in books if book['id'] == id]

        if(len(book) == 0):
            abort(404)

        return{"book": marshal(book[0], bookFields)}

    # PUT - Given an id
    def put(self, id):
        book = [book for book in books if book['id'] == id]

        if len(book) == 0:
            abort(404)

        book = book[0]

        # Loop Through all the passed agruments
        args = self.reqparse.parse_args()
        for k, v in args.items():
            # Check if the passed value is not null
            if v is not None:
                # if not, set the element in the books dict with the 'k' object to the value provided in the request.
                book[k] = v

        return{"book": marshal(book, bookFields)}

        # Delete - Given an id
    def delete(self, id):
        book = [book for book in books if book['id'] == id]

        if(len(book) == 0):
            abort(404)

        books.remove(book[0])

        return 201


class BookList(Resource):
    def __init__(self):
        self.reqparse = reqparse.RequestParser()
        self.reqparse.add_argument(
            "title", type=str, required=True, help="The title of the book must be provided", location="json")
        self.reqparse.add_argument(
            "author", type=str, required=True, help="The author of the book must be provided", location="json")
        self.reqparse.add_argument("length", type=int, required=True,
                                   help="The length of the book (in pages)", location="json")
        self.reqparse.add_argument(
            "rating", type=float, required=True, help="The rating must be provided", location="json")

    def get(self):
        return{"books": [marshal(book, bookFields) for book in books]}

    def post(self):
        args = self.reqparse.parse_args()
        book = {
            "id": books[-1]['id'] + 1 if len(books) > 0 else 1,
            "title": args["title"],
            "author": args["author"],
            "length": args["length"],
            "rating": args["rating"]
        }

        books.append(book)
        return{"book": marshal(book, bookFields)}, 201


class HelloWorld(Resource):
    def get(self):
        return {'Message': 'Welcome to ChatBot REST API'}


api.add_resource(HelloWorld, '/')

api.add_resource(ChatMessage, "/predict")

api.add_resource(BookList, "/books")
api.add_resource(Book, "/books/<int:id>")


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=port)
