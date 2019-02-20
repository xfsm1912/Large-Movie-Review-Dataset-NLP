from .base import BaseModel, BaseModel_LR, BaseModel_NB

class ReviewModel(BaseModel):
    def __init__(self):
        super().__init__()

        self.load_vec('models/tf_vec.pkl')
        self.load_model('models/svc_model.pkl')

    def predict(self, line, highlight=True):
        sentiment = super(ReviewModel, self).predict(line)

        # highlight words, hack
        if highlight:
            highlight_words = [w for w in self.preprocessing(line).split()
                               if super(ReviewModel, self).predict(w) == sentiment]
            return sentiment, highlight_words
        else:
            return sentiment

class ReviewModel_LR(BaseModel_LR):
    def __init__(self):
        super().__init__()

        self.load_vec('models/tf_vec.pkl')
        self.load_model('models/lr_model1211.pkl')


    def predict(self, line, highlight=True):
        sentiment_lr = super(ReviewModel_LR, self).predict(line)

        # highlight words, hack
        if highlight:
            highlight_words_lr = [w for w in self.preprocessing(line).split()
                               if super(ReviewModel_LR, self).predict(w) == sentiment_lr]
            return sentiment_lr, highlight_words_lr
        else:
            return sentiment_lr


class ReviewModel_NB(BaseModel_NB):
    def __init__(self):
        super().__init__()

        self.load_vec('models/tf_vec.pkl')
        self.load_model('models/mnb_model.pkl')


    def predict(self, line, highlight=True):
        sentiment_nb = super(ReviewModel_NB, self).predict(line)

        # highlight words, hack
        if highlight:
            highlight_words_nb = [w for w in self.preprocessing(line).split()
                               if super(ReviewModel_NB, self).predict(w) == sentiment_nb]
            return sentiment_nb, highlight_words_nb
        else:
            return sentiment_nb




