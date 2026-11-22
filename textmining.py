"""Gemeinsame Hilfsfunktionen für die Text-Mining-Skripte des Kurses.

Alle Skripte arbeiten mit demselben Korpus. Die Entscheidungen darüber, was als
Dokument zählt, stehen deshalb einmal hier - statt in sieben Notebooks wiederholt
zu werden und dabei unbemerkt auseinanderzulaufen. Skript 01 erklärt jede dieser
Entscheidungen Schritt für Schritt; dieses Modul ist die Kurzfassung.
"""

from __future__ import annotations

import html
import os
import re
import warnings
from pathlib import Path

import pandas as pd

# Das Parallelisierungs-Backend von scikit-learn (loky aus joblib) fragt die Anzahl
# der CPU-Kerne über `wmic` ab, das aktuelle Windows-Versionen nicht mehr
# mitliefern. Die Abfrage scheitert harmlos - es wird die Anzahl der logischen
# Kerne verwendet -, erzeugt aber mitten in der Ausgabe eine Warnung samt
# Traceback. Wir füllen den Cache von loky vorab mit genau diesem Wert und
# überspringen so die Abfrage; der Warnungsfilter deckt Versionen ab, in denen
# die Cache-Variable anders heißt.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
warnings.filterwarnings("ignore", message="Could not find the number of physical cores")
try:
    from joblib.externals.loky.backend import context as _loky_context

    if _loky_context.physical_cores_cache is None:
        _loky_context.physical_cores_cache = os.cpu_count() or 1
except ImportError:
    pass

# Die Daten stammen aus dem Nachrichtenarchiv des Projekts orf-teletext. Liegt das
# Archiv an einem anderen Ort (z. B. am Trainings-Server), setze die
# Umgebungsvariable TEXTMINING_DATA auf dessen Ordner.
ROOT = Path(os.environ.get("TEXTMINING_DATA", r"D:\projects\orf-teletext"))
SILVER = ROOT / "silver" / "items.parquet"
CACHE = Path(__file__).resolve().parent / "_cache"

# Eine Zeile je Feed: wer ihn herausgibt, woher und in welcher Sprache. Über das
# Land vergleicht Skript 03 die österreichische mit der deutschen Berichterstattung.
FEEDS = pd.DataFrame(
    [
        ("orf_news", "ORF", "AT", "de", "öffentlich-rechtlicher Sender, allgemeine Nachrichten"),
        ("derstandard_wirtschaft", "Der Standard", "AT", "de", "Tageszeitung, Wirtschaftsressort"),
        ("tagesschau", "tagesschau", "DE", "de", "öffentlich-rechtlicher Sender, allgemeine Nachrichten"),
        ("handelsblatt_finanzen", "Handelsblatt", "DE", "de", "Wirtschaftszeitung, Finanzressort"),
        ("ecb_press", "ECB", "EU", "en", "Pressemitteilungen und Reden der Zentralbank"),
    ],
    columns=["feed", "outlet", "country", "language", "profile"],
).set_index("feed")

# Feste Farben je Feed, damit ein Feed in allen Grafiken gleich aussieht.
FEED_COLOURS = {
    "orf_news": "#B33A3A",
    "tagesschau": "#2F6690",
    "derstandard_wirtschaft": "#2E7D5B",
    "ecb_press": "#946200",
    "handelsblatt_finanzen": "#6A4C93",
}

_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")


def clean_html(text: str | None) -> str:
    """Entferne Markup und HTML-Entities aus einer Feed-Beschreibung.

    derStandard stellt jeder Beschreibung ein <img>-Tag voran. Bliebe es im Text,
    würde seine URL pro Eintrag in Dutzende sinnlose "Wörter" zerlegt - und weil
    jeder Eintrag dieses Feeds eines trägt, würden sie jeden Vergleich der Feeds
    dominieren.
    """
    if not text:
        return ""
    text = _TAG_RE.sub(" ", text)
    text = html.unescape(text)
    return _SPACE_RE.sub(" ", text).strip()


def load_corpus(languages: tuple[str, ...] = ("de",), dedupe: bool = True) -> pd.DataFrame:
    """Lade eine Zeile je Nachricht, fertig für das Text Mining.

    - Liest die Silver-Tabelle, die bereits eine Zeile je Eintrag enthält
      (Bronze enthält eine Zeile je Abruf).
    - Entfernt Einträge, deren Kennung wiederverwendet wurde: Einträge, die aus dem
      Feed verschwinden und mit neuem Zeitstempel zurückkehren, sind meist eine URL,
      die ein Medium für ein tägliches Format recycelt. Ihr Text besteht also aus
      mehreren verschiedenen Meldungen.
    - Entfernt optional wiederholte Paare aus Feed und Titel: Handelsblatt
      veröffentlicht denselben Artikel unter mehreren URLs, wodurch seine Wörter
      doppelt gezählt würden.
    - Behält standardmäßig nur deutschsprachige Feeds. Der EZB-Feed ist englisch,
      und alle deutschen Werkzeuge in diesem Kurs würden ihn falsch verarbeiten.
    """
    items = pd.read_parquet(SILVER)
    composite = ~items["contiguous"] & items["republished"]
    items = items.loc[~composite]

    feeds = FEEDS.index[FEEDS["language"].isin(languages)]
    items = items.loc[items["feed"].isin(feeds)]

    corpus = pd.DataFrame(
        {
            "feed": items["feed"],
            "item_key": items["item_key"],
            "title": items["final_title"].fillna("").str.strip(),
            "description": items["final_description"].map(clean_html),
            "published": items["published"],
            "subject": items["subject"],
            "story_type": items["story_type"],
        }
    )
    corpus["outlet"] = corpus["feed"].map(FEEDS["outlet"])
    corpus["country"] = corpus["feed"].map(FEEDS["country"])
    # Titel und Beschreibung gemeinsam, für Skripte, die mehr Text je Eintrag
    # brauchen. Der Punkt dazwischen verhindert, dass das letzte Wort des Titels und
    # das erste Wort der Beschreibung ein falsches Bigramm bilden.
    corpus["text"] = corpus["title"].where(
        corpus["description"] == "", corpus["title"] + ". " + corpus["description"]
    )

    if dedupe:
        corpus = corpus.sort_values("published").drop_duplicates(["feed", "title"], keep="first")

    return corpus.sort_values(["feed", "published"]).reset_index(drop=True)


# Der ORF ordnet jede Meldung einem Ressort zu. Die fünf größten werden zu Klassen;
# die lange Liste kleiner Ressorts (Umwelt, IT, Medien, Lotto, ...) wird
# zusammengefasst, weil sich eine Klasse mit einem Dutzend Beispielen weder
# zuverlässig lernen noch evaluieren lässt.
SUBJECTS = ["Ausland", "Inland", "Wirtschaft", "Chronik", "Kultur"]
OTHER = "Sonstiges"


def subject_task(test_share: float = 0.25) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Die Klassifikationsaufgabe der Skripte 05-07: Sage das ORF-Ressort aus dem Titel vorher.

    Die Aufteilung erfolgt zeitlich, nicht zufällig: Der jüngste Anteil
    `test_share` der Einträge wird zurückgehalten. Nachrichten wiederholen sich -
    Folgemeldungen, Korrekturen, fast identische Schlagzeilen an aufeinanderfolgenden
    Tagen -, sodass eine zufällige Aufteilung Beinahe-Kopien der Testdaten in die
    Trainingsdaten bringen und die Genauigkeit überschätzen würde. Alle Skripte
    verwenden diese eine Aufteilung; nur deshalb sind ihre Ergebnisse vergleichbar.
    """
    corpus = load_corpus()
    orf = corpus[(corpus["feed"] == "orf_news") & corpus["subject"].notna()].copy()
    orf["label"] = orf["subject"].where(orf["subject"].isin(SUBJECTS), OTHER)
    orf = orf.sort_values("published").reset_index(drop=True)
    cut = int(len(orf) * (1 - test_share))
    return orf.iloc[:cut].reset_index(drop=True), orf.iloc[cut:].reset_index(drop=True)


def parse_corpus(nlp, texts: list[str], name: str, batch_size: int = 128):
    """Lass spaCy einmal über die Texte laufen und speichere das Ergebnis auf der Festplatte.

    Das Parsen einiger tausend Einträge dauert lange genug, um zu stören, wenn es in
    jedem Skript wiederholt wird. DocBin ist das Binärformat von spaCy für genau
    diesen Zweck. Der Cache-Schlüssel enthält den Modellnamen und die Anzahl der
    Texte; ein neu aufgebauter Korpus oder ein anderes Modell führt daher zu einem
    neuen Parse statt zu einem veralteten.
    """
    from spacy.tokens import DocBin

    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"{name}-{nlp.meta['name']}-{len(texts)}.spacy"
    if path.exists():
        docs = list(DocBin().from_disk(path).get_docs(nlp.vocab))
        if [d.text for d in docs] == list(texts):
            return docs
    docs = list(nlp.pipe(texts, batch_size=batch_size))
    DocBin(store_user_data=False, docs=docs).to_disk(path)
    return docs
