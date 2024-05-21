from sklearn.feature_extraction.text import TfidfVectorizer

#Datei mit Beispieltext 3 laden
article2 = open("/Users/dgromann/Library/CloudStorage/GoogleDrive-dagmar.gromann@gmail.com/My Drive/UniWien/Teaching/Einführung_Programmieren_Translator_innen_SS2024/Text2.txt")
article3 = open("/Users/dgromann/Library/CloudStorage/GoogleDrive-dagmar.gromann@gmail.com/My Drive/UniWien/Teaching/Einführung_Programmieren_Translator_innen_SS2024/Text3.txt")

# Hier finden Sie die Scikit-Learn Dokumentation:
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
def tf_idf_sklearn(corpus):

  # Erst wird ein Objekt dieser Klasse erzeugt
  vectorizer = TfidfVectorizer()

  # Mit der Funktion fit_transform werden die TF-IDF-Werte angesprochen
  tf_idf_scores = vectorizer.fit_transform(corpus)

  # Mit dieser Funktion kann die Position des Terms im Vokabular abgerufen werden
  terms = vectorizer.get_feature_names_out()

  tf_idf_dict = {}
  for term in terms:
    tf_idf_dict[term] = tf_idf_scores[0, vectorizer.vocabulary_[term]]

  return tf_idf_dict

if __name__ == '__main__':
    text1 = article2.read()
    text2 = article3.read()
    corpus = [text1, text2]

    result = tf_idf_sklearn(corpus)

    # Aufgabe: Geben Sie nur Wörter mit TF-IDF-Werten über Null aus
    print(result)