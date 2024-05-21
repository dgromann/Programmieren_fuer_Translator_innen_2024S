import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

article2 = open("Dateipfad/Text2.txt")
article3 = open("Dateipfad/Text3.txt")

spacy.cli.download("de_core_news_lg")

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


'''
Aufgabe: 
Führen Sie die folgenden Vorverarbeitungsschritte durch: 
1. Tokenisierung 
2. POS-Tagging und Enfernen der angegebenen POS-Tags
3. Lemmatisierung
'''
def preprocess(corpus):
    pos_to_be_removed = ['ADV', 'PRON', 'CCONJ', 'PUNCT', 'PART', 'DET', 'ADP', 'SPACE']

    nlp = spacy.load("de_core_news_lg")

    # Hier sollte Ihr Code stehen

'''
Aufgabe: 
Erstellen Sie eine Funnktion, die nur die Named Entities, also 
Eigennamen aus den Dateien zurückgibt
'''
def ner(corpus):


if __name__ == '__main__':
    text1 = article2.read()
    text2 = article3.read()
    corpus = [text1, text2]

    # Aufgabe:
    # Verarbeiten Sie die Texte erst vor mit preprocess,
    # bevor Sie die Texte der Funktion übergeben
    result = tf_idf_sklearn(corpus)

    # Aufgabe:
    # Extrahieren Sie alle NERs mithilfe der Funktion ner und lassen Sie sich die
    # TF-IDF-Werte dieser ausgeben

    # Aufgabe:
    # Speichern Sie das Ergebnisse als Datei
