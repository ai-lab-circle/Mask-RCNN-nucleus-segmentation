from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from scipy.spatial import distance

X = CountVectorizer().fit_transform(docs)
X = TfidfTransformer(use_idf=False).fit_transform(X)
print (X.shape) #prints (100, 1760)
distance.pdist(X, metric='cosine')