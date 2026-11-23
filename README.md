# Text Mining

Der Text-Mining-Teil des Seminars, aufbauend auf dem Data-Mining-Teil in `intro-data-mining-de`. Er besteht aus sieben Skripten und arbeitet mit Schlagzeilen und Teasern zweier Tageszeitungen aus demselben Jahr (Juni 2015 bis Mai 2016): **Der Standard** aus Wien und die **Süddeutsche Zeitung** aus München. Beide stammen aus frei verfügbaren Forschungskorpora. Jedes Skript stellt eine Familie von Methoden vor, wendet sie auf das Korpus an und überprüft die Ergebnisse anhand der zugrunde liegenden Texte.

Die ersten sechs Skripte verwenden klassische, nachvollziehbare Methoden. Ein großes Sprachmodell (LLM) kommt erst in Skript 07 vor und wird dort unter gleichen Bedingungen mit allen vorher entwickelten Verfahren verglichen.

| # | Skript | Methoden | Pakete |
|---|---|---|---|
| 01 | [Text als Daten](01-text-mining-text-as-data.ipynb) | Dokumente und Bereinigung, Tokenisierung, Normalisierung, Stoppwörter, Zipfsches Gesetz, N-Gramme, Komposita | pandas, re, NLTK |
| 02 | [Linguistische Annotation](02-text-mining-linguistic-annotation.ipynb) | spaCy-Pipelines, Lemmata, Wortarten, Morphologie, Dependenzparsing, Named Entities, regelbasiertes Matching | spaCy |
| 03 | [Korpora vergleichen](03-text-mining-comparing-corpora.ipynb) | Dokument-Term-Matrix, TF-IDF, Keyness (G², Log Ratio), PMI-Kollokationen, Konkordanzen, Begriffe im Zeitverlauf | scikit-learn |
| 04 | [Ähnlichkeit, Clustering und Topic Models](04-text-mining-similarity-and-topics.ipynb) | Kosinus-Ähnlichkeit, Zuordnung derselben Geschichte in zwei Zeitungen, LSA, k-means, Topic Models mit NMF und LDA, Prüfung gegen die Ressorts | scikit-learn |
| 05 | [Textklassifikation](05-text-mining-classification.ipynb) | Baselines, zeitliche Aufteilung und Data Leakage, Naive Bayes / logistische Regression / SVM, Zeichen-N-Gramme, Evaluation, Fehleranalyse | scikit-learn |
| 06 | [Embeddings](06-text-mining-embeddings.ipynb) | Wortvektoren, Sentence Transformer, semantische und sprachübergreifende Suche, Embeddings als Merkmale | spaCy, sentence-transformers, UMAP |
| 07 | [Ein LLM als Klassifikator](07-text-mining-llm-classification.ipynb) | Zero-Shot- und Few-Shot-Prompting, strukturierte Ausgabe, Kosten und Validität | anthropic |

Die Skripte 05, 06 und 07 bearbeiten dieselbe Aufgabe - das Ressort eines Standard-Artikels aus seiner Schlagzeile vorhersagen (die sechs größten Ressorts und *Sonstiges*) - mit derselben zeitlichen Aufteilung in Trainings- und Testdaten. Ihre Ergebnisse sind daher direkt vergleichbar.

## Daten

| Zeitung | Quelle | Umfang im Kurs | Lizenz |
|---|---|---|---|
| Der Standard (AT) | One Million Posts Corpus, OFAI (Schabus et al. 2017): Artikel mit Titel, Zeitstempel, Ressort-Pfad und HTML-Text | rund 12.000 Artikel | CC BY-NC-SA 4.0 |
| Süddeutsche Zeitung (DE) | MLSUM, deutscher Teil (Scialom et al. 2020), über die Kopie im GEM-Benchmark auf Hugging Face: Titel, Teaser, Text, Ressort, Monat | rund 31.000 Artikel aus demselben Zeitraum | nur für nicht-kommerzielle Forschung und Lehre |

Beide Lizenzen erlauben die Nutzung in der Lehre, nicht aber eine kommerzielle Verwendung. Wer die Daten oder daraus abgeleitete Ergebnisse weitergibt, nennt die Quellen:

- Schabus, D., Skowron, M., Trapp, M. (2017): One Million Posts: A Data Set of German Online Discussions. *Proceedings of SIGIR 2017*, 1241-1244. https://ofai.github.io/million-post-corpus/
- Scialom, T., Dray, P.-A., Lamprier, S., Piwowarski, B., Staiano, J. (2020): MLSUM: The Multilingual Summarization Corpus. *Proceedings of EMNLP 2020*, 8051-8067. https://github.com/ThomasScialom/MLSUM

**Download.** Die Daten liegen nicht im Repository. `textmining.py` lädt sie beim ersten Aufruf von `load_corpus()` herunter (rund 1 GB: das One Million Posts Corpus als SQLite-Datenbank und der deutsche Trainingsteil von MLSUM als Parquet), behält nur die Artikel aus dem gemeinsamen Zeitraum und speichert sie als zwei Parquet-Dateien (zusammen knapp 100 MB) im Ordner `data/` neben den Skripten. Die Rohdateien werden danach gelöscht. Der Schritt lässt sich auch vorab ausführen:

```bash
python -c "import textmining; textmining.prepare_data()"
```

Sollen die Daten an einem anderen Ort liegen (etwa auf einem gemeinsamen Laufwerk), setze die Umgebungsvariable `TEXTMINING_DATA` auf diesen Ordner, bevor du Jupyter startest. `data/` ist in `.gitignore` eingetragen.

Was als Dokument zählt, entscheidet `load_corpus()` in mehreren Schritten (gemeinsamer Zeitraum, Community- und Serviceseiten, Platzhaltertitel, wiederkehrende Formate, Doubletten, HTML, Spuren des CSV-Exports, Wahl des Teasers). Skript 01 erklärt jede dieser Entscheidungen.

## Folien

Die Präsentation zum Seminar liegt als [`text-mining.odp`](text-mining.odp) im OpenDocument-Format vor und lässt sich mit LibreOffice Impress oder PowerPoint öffnen. Sie führt in das Text Mining ein und stellt jedes Skript mit einer Folie zu den Methoden und einem Code-Beispiel samt Ergebnissen vor. Die Zahlen, Beispiele und Grafiken auf den Folien stammen aus einem Lauf der Skripte auf dem hier beschriebenen Korpus; sie ändern sich nicht mit, wenn die Skripte verändert werden.

## Ausführen

Die Skripte brauchen Python 3.10 oder neuer. Lege eine virtuelle Umgebung an (oder verwende eine vorhandene gemeinsame Umgebung) und installiere zusätzlich zu den üblichen wissenschaftlichen Paketen:

```bash
python -m pip install pandas matplotlib spacy nltk pyarrow scikit-learn sentence-transformers torch umap-learn anthropic jupyterlab
```

```bash
python -m spacy download de_core_news_sm
```

```bash
python -m spacy download de_core_news_md
```

```bash
python -m spacy download de_core_news_lg
```

Starte dann Jupyter aus diesem Ordner, mit dem Python der Umgebung:

```bash
python -m jupyter lab
```

Alternativ öffne die Notebooks in Positron oder VS Code mit der Umgebung als Interpreter. Sie sind **ohne Ausgaben** gespeichert; führe alle Zellen aus, um sie zu befüllen. Ohne GPU dauert ein vollständiger Durchlauf aller Skripte rund eine halbe Stunde; der erste Lauf braucht zusätzlich den Download der Daten (Skript 01) und das einmalige Parsen mit spaCy (Skripte 02 und 04), deren Ergebnisse zwischengespeichert werden.

Skript 06 lädt beim ersten Aufruf ein Sentence-Transformer-Modell (etwa 0,5 GB) vom Hugging Face Hub. Skript 07 ruft die API von Anthropic nur auf, wenn in seiner ersten Code-Zelle `RUN_LLM = True` gesetzt ist, und benötigt dann `ANTHROPIC_API_KEY` als Umgebungsvariable. Mit `RUN_LLM = False` läuft es vollständig durch, ohne Kosten zu verursachen.

## Aufbau

- [`textmining.py`](textmining.py) enthält, was alle Skripte gemeinsam nutzen: Download und Laden des Korpus mit den Entscheidungen darüber, was ein Dokument ist (Schritt für Schritt erklärt in Skript 01), Metadaten und Farben der Zeitungen, die gemeinsamen Ressortgruppen, einen zwischengespeicherten spaCy-Parse und die Aufteilung für die Klassifikation.
- [`text-mining.odp`](text-mining.odp) enthält die Folien zum Seminar (siehe oben).
- `data/` enthält die heruntergeladenen Daten (nicht im Repository, siehe oben).
- `_cache/` enthält spaCy-Parses und gespeicherte LLM-Ergebnisse. Das sind abgeleitete Daten, die gefahrlos gelöscht werden können. Wird eine LLM-Ergebnisdatei gelöscht, kostet der nächste Lauf mit `RUN_LLM = True` erneut Geld.
