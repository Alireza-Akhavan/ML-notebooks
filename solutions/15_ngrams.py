text = zen.split("\n")
for n in [2, 3, 4]:
    cv = CountVectorizer(ngram_range=(n, n)).fit(text)
    counts = cv.transform(text)
    most_common = np.argmax(counts.sum(axis=0))
    print("most common %d-gram: %s" % (n, cv.get_feature_names()[most_common]))
