from simalign import SentenceAligner

'''
Vergleichen Sie die visuelle Darstellung der beiden Sätze in dieser Online-Demo mit der Ausgabe: https://simalign.cis.lmu.de/ 
mwmf = Match, inter = ArgMax, itermax = IterMax

Eine interessante Methode zur Alignierung von Sätzen ist BertAlign: https://github.com/bfsujason/bertalign/tree/main
Dazu können Sie sich das GitHub sowie den darin enthaltenen Google Colab-Link anschauen 
'''
def find_word_alignment(source_sentence, target_sentence):
    simalign_bert = SentenceAligner(model="bert-base-multilingual-cased", token_type="word")
    alignments = simalign_bert.get_word_aligns(source_sentence, target_sentence)

    return alignments

if __name__ == '__main__':
    source_sentence = "Sir Nils Olav III. was knighted by the norwegian king ."
    target_sentence = "Nils Olav der Dritte wurde vom norwegischen König zum Ritter geschlagen ."

    ''' 
    Aufgabe: 
    Schreiben Sie die Funktion "find_word_alignment so um, dass nicht die Indizes der Wörter sondern 
    die miteinander alignierten Wörter direkt zurückgegeben werden. 
    Hier treffen Sie auf einen neuen Datentyp namens Tuple, z. B. (3, 2). Tuples können ebenso wie Listen indiziert werden. 
    Also tuple = (0, 1) kann angesprochen werden als tuple[0] und tuple[1], um die darin enthaltenen Werte zu erhalten. 
    '''
    result = find_word_alignment(source_sentence, target_sentence)
