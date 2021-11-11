!pip install gensim
!pip install python-Levenshtein

import pandas as pd
import gensim

df = pd.read_json("reviews_Cell_Phones_and_Accessories_5.json", lines=True)
df.head()

df.shape

df.reviewText[193321]

gensim.utils.simple_preprocess('Love the polka dots! This is a very thin and lightweight and flexible cover for Galaxy S5.  Very cute case.')

review_text = df.reviewText.apply(gensim.utils.simple_preprocess)
review_text

model = gensim.models.Word2Vec(
    window=10,
    # window is the amount of words to each side of the center word
    min_count=2,
    # min count is number of words in a sentence in order for us to train
    workers=3
    # workers is the number of cpu threads your computer has 
)

model.build_vocab(review_text, progress_per=1000)
# progress_per was a parameter chosen based on the documentation for genism and the build_vocab function

model.epochs
# epochs is how many times you want to iterate through the entire dataset

model.train(review_text, total_examples=model.corpus_count, epochs=model.epochs)

model.save("./word2vec-amazon-cell-accessories-reviews-short.model")

model.wv.most_similar("bad")

model.wv.similarity(w1="great", w2="samsung")
