# Text Mining

Der Text-Mining-Teil des Seminars, aufbauend auf dem Data-Mining-Teil in `D:\projects\intro-data-mining-de`. Er besteht aus sieben Skripten und arbeitet mit einem Archiv deutschsprachiger Nachrichten: Schlagzeilen und Teaser von ORF, Der Standard, tagesschau und Handelsblatt. Jedes Skript stellt eine Familie von Methoden vor, wendet sie auf das Archiv an und überprüft die Ergebnisse anhand der zugrunde liegenden Texte.

Die ersten sechs Skripte verwenden klassische, nachvollziehbare Methoden. Ein großes Sprachmodell (LLM) kommt erst in Skript 07 vor und wird dort unter gleichen Bedingungen mit allen vorher entwickelten Verfahren verglichen.

| # | Skript | Methoden | Pakete |
|---|---|---|---|
| 01 | [Text als Daten](01-text-mining-text-as-data.ipynb) | Dokumente und Deduplizierung, Tokenisierung, Normalisierung, Stoppwörter, Zipfsches Gesetz, N-Gramme, Komposita | pandas, re, NLTK |
| 02 | [Linguistische Annotation](02-text-mining-linguistic-annotation.ipynb) | spaCy-Pipelines, Lemmata, Wortarten, Morphologie, Dependenzparsing, Named Entities, regelbasiertes Matching | spaCy |
| 03 | [Korpora vergleichen](03-text-mining-comparing-corpora.ipynb) | Dokument-Term-Matrix, TF-IDF, Keyness (G², Log Ratio), PMI-Kollokationen, Konkordanzen, Begriffe im Zeitverlauf | scikit-learn |
| 04 | [Ähnlichkeit, Clustering und Topic Models](04-text-mining-similarity-and-topics.ipynb) | Kosinus-Ähnlichkeit, Zuordnung derselben Meldung über Medien hinweg, LSA, k-means, Topic Models mit NMF und LDA | scikit-learn |
| 05 | [Textklassifikation](05-text-mining-classification.ipynb) | Baselines, zeitliche Aufteilung und Data Leakage, Naive Bayes / logistische Regression / SVM, Zeichen-N-Gramme, Evaluation, Fehleranalyse | scikit-learn |
| 06 | [Embeddings](06-text-mining-embeddings.ipynb) | Wortvektoren, Sentence Transformer, semantische und sprachübergreifende Suche, Embeddings als Merkmale | spaCy, sentence-transformers, UMAP |
| 07 | [Ein LLM als Klassifikator](07-text-mining-llm-classification.ipynb) | Zero-Shot- und Few-Shot-Prompting, strukturierte Ausgabe, Kosten und Validität | anthropic |

Die Skripte 05, 06 und 07 bearbeiten dieselbe Aufgabe - das ORF-Ressort aus einer Schlagzeile vorhersagen - mit derselben Aufteilung in Trainings- und Testdaten. Ihre Ergebnisse sind daher direkt vergleichbar.

## Ausführen

Starte Jupyter mit der gemeinsamen Umgebung aus diesem Ordner:

```bash
D:\python-shared\Scripts\python -m jupyter lab
```

Alternativ öffne die Notebooks in Positron mit `D:\python-shared` als Interpreter. Sie sind **ohne Ausgaben** gespeichert; führe alle Zellen aus, um sie zu befüllen.

Alles Nötige ist in `D:\python-shared` installiert. Wird die Umgebung neu aufgebaut, braucht der Kurs zusätzlich zu den üblichen wissenschaftlichen Paketen:

```bash
D:\python-shared\Scripts\python -m pip install spacy nltk pyarrow scikit-learn sentence-transformers torch umap-learn anthropic jupyterlab
```

```bash
D:\python-shared\Scripts\python -m spacy download de_core_news_sm
```

```bash
D:\python-shared\Scripts\python -m spacy download de_core_news_md
```

```bash
D:\python-shared\Scripts\python -m spacy download de_core_news_lg
```

Skript 06 lädt beim ersten Aufruf ein Sentence-Transformer-Modell (etwa 0,5 GB) vom Hugging Face Hub. Skript 07 ruft die API von Anthropic nur auf, wenn in seiner ersten Code-Zelle `RUN_LLM = True` gesetzt ist, und benötigt dann `ANTHROPIC_API_KEY` als Umgebungsvariable. Mit `RUN_LLM = False` läuft es vollständig durch, ohne Kosten zu verursachen.

## Daten

Der Korpus stammt aus dem Nachrichtenarchiv des Projekts orf-teletext. `textmining.py` liest ihn aus `D:\projects\orf-teletext\silver\items.parquet` (Skript 01 zusätzlich den Ordner `bronze`). Liegt das Archiv anderswo, setze die Umgebungsvariable `TEXTMINING_DATA` auf dessen Ordner, bevor du Jupyter startest.

Wird die Silver-Tabelle in orf-teletext neu aufgebaut, verwenden alle Skripte automatisch die neuen Daten. Die Zahlen in den Ausgaben ändern sich dann; der Text beschreibt Muster statt konkreter Werte und bleibt daher gültig, während das Archiv wächst.

## Aufbau

- [`textmining.py`](textmining.py) enthält, was alle Skripte gemeinsam nutzen: das Laden des Korpus mit den Entscheidungen darüber, was ein Dokument ist (Schritt für Schritt erklärt in Skript 01), Metadaten und Farben der Feeds, einen zwischengespeicherten spaCy-Parse und die Aufteilung für die Klassifikation.
- `_cache/` enthält spaCy-Parses und gespeicherte LLM-Ergebnisse. Das sind abgeleitete Daten, die gefahrlos gelöscht werden können. Wird eine LLM-Ergebnisdatei gelöscht, kostet der nächste Lauf mit `RUN_LLM = True` erneut Geld.
