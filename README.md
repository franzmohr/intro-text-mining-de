# Text Mining

Der Text-Mining-Teil des Seminars, aufbauend auf dem Data-Mining-Teil in `intro-data-mining-de`. Er besteht aus einem Notebook mit Vorbereitungen und neun Skripten und arbeitet mit zwei Tageszeitungen aus demselben Jahr (Juni 2015 bis Mai 2016): **Der Standard** aus Wien und die **Süddeutsche Zeitung** aus München. Beide stammen aus frei verfügbaren Forschungskorpora. Jedes Skript stellt eine Familie von Methoden vor, wendet sie auf das Korpus an und überprüft die Ergebnisse anhand der zugrunde liegenden Texte.

Die ersten sechs Skripte verwenden klassische, nachvollziehbare Methoden. Ein großes Sprachmodell (LLM) kommt erst in Skript 07 vor und wird dort unter gleichen Bedingungen mit allen vorher entwickelten Verfahren verglichen. Die Skripte 08 und 09 greifen Aufgaben auf, die Han, Pei und Tong (2023, *Data Mining*, 4. Aufl., Kap. 12.1) als Forschungsfronten des Text Mining beschreiben: Informationsextraktion, Taxonomien, Argument Mining und Textzusammenfassung.

| # | Skript | Methoden | Pakete |
|---|---|---|---|
| 00 | [Vorbereitungen](00-preparations.ipynb) | Pakete, spaCy-Modelle und Daten installieren, API-Schlüssel prüfen | pip |
| 01 | [Text als Daten](01-text-mining-text-as-data.ipynb) | Dokumente und Bereinigung, Tokenisierung, Normalisierung, Stoppwörter, Zipfsches Gesetz, N-Gramme, Komposita | pandas, re, NLTK |
| 02 | [Linguistische Annotation](02-text-mining-linguistic-annotation.ipynb) | spaCy-Pipelines, Lemmata, Wortarten, Morphologie, Dependenzparsing, Named Entities, regelbasiertes Matching | spaCy |
| 03 | [Korpora vergleichen](03-text-mining-comparing-corpora.ipynb) | Dokument-Term-Matrix, TF-IDF, Keyness (G², Log Ratio), PMI-Kollokationen, Konkordanzen, Begriffe im Zeitverlauf | scikit-learn |
| 04 | [Ähnlichkeit, Clustering und Topic Models](04-text-mining-similarity-and-topics.ipynb) | Kosinus-Ähnlichkeit, Zuordnung derselben Geschichte in zwei Zeitungen, LSA, k-means, Topic Models mit NMF und LDA, Prüfung gegen die Ressorts | scikit-learn |
| 05 | [Textklassifikation](05-text-mining-classification.ipynb) | Baselines, zeitliche Aufteilung und Data Leakage, Naive Bayes / logistische Regression / SVM, Zeichen-N-Gramme, Evaluation, Fehleranalyse | scikit-learn |
| 06 | [Embeddings](06-text-mining-embeddings.ipynb) | Wortvektoren, eigene Wortvektoren mit word2vec (CBOW, Skip-gram) und fastText, Sentence Transformer, statische und kontextuelle Embeddings, semantische und sprachübergreifende Suche, Embeddings als Merkmale, schwache Überwachung und Selbsttraining | spaCy, gensim, sentence-transformers, UMAP |
| 07 | [Ein LLM als Klassifikator](07-text-mining-llm-classification.ipynb) | Zero-Shot- und Few-Shot-Prompting, strukturierte Ausgabe, Kosten und Validität | anthropic |
| 08 | [Informationsextraktion und Argument Mining](08-text-mining-information-extraction.ipynb) | Rollen und Relationen aus dem Dependenzbaum, Ereignisse mit Ort und Zeit, Taxonomien mit Hearst-Mustern, argumentierende Leserkommentare erkennen (mit der Länge als Störfaktor), Behauptung und Prämisse | spaCy, scikit-learn |
| 09 | [Textzusammenfassung](09-text-mining-summarization.ipynb) | ROUGE, Prüfung der Referenzen, extraktive Verfahren (Lead, Zentroide, TextRank, Orakel), abstraktive und Extract-then-Rewrite-Zusammenfassung mit Claude und Mistral, erfundene Fakten | spaCy, sentence-transformers, anthropic, mistralai |

Die Skripte 05, 06 und 07 bearbeiten dieselbe Aufgabe - das Ressort eines Standard-Artikels aus seiner Schlagzeile vorhersagen (die sechs größten Ressorts und *Sonstiges*) - mit derselben zeitlichen Aufteilung in Trainings- und Testdaten. Ihre Ergebnisse sind daher direkt vergleichbar.

## Daten

| Quelle | Inhalt | Umfang im Kurs | Lizenz |
|---|---|---|---|
| Der Standard (AT): One Million Posts Corpus, OFAI (Schabus et al. 2017) | Artikel mit Titel, Zeitstempel, Ressort-Pfad und HTML-Text; Leserkommentare, von der Redaktion annotiert (Argumente, Stimmung, Off-Topic, ...) | rund 12.000 Artikel, rund 11.800 annotierte Kommentare | CC BY-NC-SA 4.0 |
| Süddeutsche Zeitung (DE): MLSUM, deutscher Teil (Scialom et al. 2020), über die Kopie im GEM-Benchmark auf Hugging Face | Titel, Teaser, Text, Ressort, Monat | rund 31.000 Artikel aus demselben Zeitraum | nur für nicht-kommerzielle Forschung und Lehre |

Die meisten Skripte arbeiten mit Schlagzeile und Teaser (`tm.load_corpus()`). Für eigene Wortvektoren (06), Taxonomien und Argument Mining (08) und die Zusammenfassung (09) liefert `tm.load_bodies()` die vollen Artikeltexte, für das Argument Mining `tm.load_posts()` die annotierten Leserkommentare.

Beide Lizenzen erlauben die Nutzung in der Lehre, nicht aber eine kommerzielle Verwendung. Wer die Daten oder daraus abgeleitete Ergebnisse weitergibt, nennt die Quellen:

- Schabus, D., Skowron, M., Trapp, M. (2017): One Million Posts: A Data Set of German Online Discussions. *Proceedings of SIGIR 2017*, 1241-1244. https://ofai.github.io/million-post-corpus/
- Scialom, T., Dray, P.-A., Lamprier, S., Piwowarski, B., Staiano, J. (2020): MLSUM: The Multilingual Summarization Corpus. *Proceedings of EMNLP 2020*, 8051-8067. https://github.com/ThomasScialom/MLSUM

**Download.** Die Daten liegen nicht im Repository. `textmining.py` lädt sie beim ersten Aufruf von `load_corpus()` herunter (rund 1 GB: das One Million Posts Corpus als SQLite-Datenbank und der deutsche Trainingsteil von MLSUM als Parquet), behält nur die Artikel aus dem gemeinsamen Zeitraum und die annotierten Kommentare und speichert sie als drei Parquet-Dateien (zusammen knapp 100 MB) im Ordner `data/` neben den Skripten. Die Rohdateien werden danach gelöscht. Der Schritt lässt sich vorab ausführen, mit Schritt 4 in [`00-preparations.ipynb`](00-preparations.ipynb) oder im Terminal:

```bash
python -c "import textmining; textmining.prepare_data()"
```

Sollen die Daten an einem anderen Ort liegen (etwa auf einem gemeinsamen Laufwerk), setze die Umgebungsvariable `TEXTMINING_DATA` auf diesen Ordner, bevor du Jupyter startest. `data/` ist in `.gitignore` eingetragen.

Was als Dokument zählt, entscheidet `load_corpus()` in mehreren Schritten (gemeinsamer Zeitraum, Community- und Serviceseiten, Platzhaltertitel, wiederkehrende Formate, Doubletten, HTML, Spuren des CSV-Exports, Wahl des Teasers). Skript 01 erklärt jede dieser Entscheidungen.

## Folien

Die Präsentation zum Seminar liegt als [`text-mining.odp`](text-mining.odp) im OpenDocument-Format vor und lässt sich mit LibreOffice Impress oder PowerPoint öffnen. Sie führt in das Text Mining ein und stellt jedes Skript mit einer Folie zu den Methoden und einem Code-Beispiel samt Ergebnissen vor. Die Zahlen, Beispiele und Grafiken auf den Folien stammen aus einem Lauf der Skripte auf dem hier beschriebenen Korpus; sie ändern sich nicht mit, wenn die Skripte verändert werden.

## Ausführen

Die Skripte brauchen Python 3.10 oder neuer. Lege eine virtuelle Umgebung an (oder verwende eine vorhandene gemeinsame Umgebung), öffne [`00-preparations.ipynb`](00-preparations.ipynb) mit dieser Umgebung und führe es einmal aus. Es installiert die Pakete und die spaCy-Modelle, lädt die Daten herunter und prüft die API-Schlüssel. Wer lieber im Terminal arbeitet:

```bash
python -m pip install anthropic gensim jupyterlab matplotlib mistralai nltk numpy pandas pyarrow scikit-learn sentence-transformers spacy torch umap-learn
```

```bash
python -m spacy download de_core_news_sm
python -m spacy download de_core_news_md
python -m spacy download de_core_news_lg
```

Starte dann Jupyter aus diesem Ordner, mit dem Python der Umgebung:

```bash
python -m jupyter lab
```

Alternativ öffne die Notebooks in Positron oder VS Code mit der Umgebung als Interpreter. Sie sind **ohne Ausgaben** gespeichert; führe alle Zellen aus, um sie zu befüllen. Ohne GPU dauert ein vollständiger Durchlauf aller Skripte rund eine Stunde; der erste Lauf braucht zusätzlich den Download der Daten, das einmalige Parsen mit spaCy (Skripte 02 und 04) und das Training der Wortvektoren (Skript 06), deren Ergebnisse zwischengespeichert werden.

Die Skripte 06, 07 und 09 laden beim ersten Aufruf ein Sentence-Transformer-Modell (etwa 0,5 GB) vom Hugging Face Hub.

**LLMs und API-Schlüssel.** Skript 07 ruft die API von Anthropic auf, Skript 09 die APIs von Anthropic und Mistral. Die Schlüssel stehen in der Datei `api-keys.env` in diesem Ordner (eine Vorlage ohne Schlüssel liegt als [`api-keys.example.env`](api-keys.example.env) daneben); `tm.load_api_keys()` liest sie. `api-keys.env` ist in `.gitignore` eingetragen und wird nie committet. Beide Skripte rufen eine API nur auf, wenn in ihrer ersten Code-Zelle `RUN_LLM = True` gesetzt ist. Mit `RUN_LLM = False` laufen sie vollständig durch, ohne Kosten zu verursachen, und verwenden gespeicherte Ergebnisse aus `_cache/`, falls vorhanden.

## Aufbau

- [`textmining.py`](textmining.py) enthält, was alle Skripte gemeinsam nutzen: Download und Laden des Korpus mit den Entscheidungen darüber, was ein Dokument ist (Schritt für Schritt erklärt in Skript 01), die vollen Artikeltexte und die annotierten Leserkommentare, Metadaten und Farben der Zeitungen, die gemeinsamen Ressortgruppen, einen zwischengespeicherten spaCy-Parse, die Aufteilung für die Klassifikation und das Lesen der API-Schlüssel.
- [`text-mining.odp`](text-mining.odp) enthält die Folien zum Seminar (siehe oben).
- [`api-keys.example.env`](api-keys.example.env) ist die Vorlage für `api-keys.env` (siehe oben).
- `data/` enthält die heruntergeladenen Daten (nicht im Repository, siehe oben).
- `_cache/` enthält spaCy-Parses, trainierte Wortvektoren und gespeicherte LLM-Ergebnisse. Das sind abgeleitete Daten, die gefahrlos gelöscht werden können. Wird eine LLM-Ergebnisdatei gelöscht, kostet der nächste Lauf mit `RUN_LLM = True` erneut Geld.
