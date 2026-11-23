"""Gemeinsame Hilfsfunktionen für die Text-Mining-Skripte des Kurses.

Alle Skripte arbeiten mit demselben Korpus: Schlagzeilen und Teaser zweier
Tageszeitungen aus demselben Jahr (Juni 2015 bis Mai 2016),

- **Der Standard** (Wien, AT) aus dem One Million Posts Corpus (Schabus et al. 2017,
  CC BY-NC-SA 4.0) und
- **Süddeutsche Zeitung** (München, DE) aus dem MLSUM-Korpus (Scialom et al. 2020,
  nur für nicht-kommerzielle Forschung und Lehre).

Die Entscheidungen darüber, was als Dokument zählt, stehen einmal hier - statt in
sieben Notebooks wiederholt zu werden und dabei unbemerkt auseinanderzulaufen.
Skript 01 erklärt jede dieser Entscheidungen Schritt für Schritt; dieses Modul ist
die Kurzfassung.

Die Rohdaten werden beim ersten Aufruf von `load_corpus()` heruntergeladen und in
zwei kleine Parquet-Dateien umgewandelt (`prepare_data()`, etwa 1 GB Download,
einige Minuten). Standardmäßig landen sie im Ordner `data/` neben diesem Modul;
die Umgebungsvariable TEXTMINING_DATA legt einen anderen Ordner fest.
"""

from __future__ import annotations

import html
import os
import re
import shutil
import sqlite3
import tarfile
import urllib.request
import warnings
from pathlib import Path

import pandas as pd

# Das Parallelisierungs-Backend von scikit-learn (loky aus joblib) fragt unter
# Windows die Anzahl der CPU-Kerne über `wmic` ab, das aktuelle Windows-Versionen
# nicht mehr mitliefern. Die Abfrage scheitert harmlos, erzeugt aber mitten in der
# Ausgabe eine Warnung samt Traceback. Wir füllen den Cache von loky vorab mit der
# Anzahl der logischen Kerne und überspringen so die Abfrage; unter Linux und macOS
# ändert das nichts.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
warnings.filterwarnings("ignore", message="Could not find the number of physical cores")
try:
    from joblib.externals.loky.backend import context as _loky_context

    if _loky_context.physical_cores_cache is None:
        _loky_context.physical_cores_cache = os.cpu_count() or 1
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Orte
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("TEXTMINING_DATA", HERE / "data"))
STANDARD_FILE = DATA / "derstandard.parquet"
SZ_FILE = DATA / "sueddeutsche.parquet"
CACHE = HERE / "_cache"

# Quellen der Rohdaten
OMPC_URL = "https://github.com/OFAI/million-post-corpus/releases/download/v1.0.0/million_post_corpus.tar.bz2"
MLSUM_URLS = [  # MLSUM Deutsch, Trainingsteil, als Parquet-Kopie des GEM-Benchmarks
    "https://huggingface.co/api/datasets/GEM/mlsum/parquet/de/train/0.parquet",
    "https://huggingface.co/api/datasets/GEM/mlsum/parquet/de/train/1.parquet",
]

# Der gemeinsame Zeitraum: Das One Million Posts Corpus umfasst die Artikel von
# Juni 2015 bis Mai 2016. Aus MLSUM behalten wir dieselben zwölf Monate.
START, END = pd.Timestamp("2015-06-01"), pd.Timestamp("2016-06-01")

# ---------------------------------------------------------------------------
# Metadaten der beiden Zeitungen
# ---------------------------------------------------------------------------

OUTLETS = pd.DataFrame(
    [
        ("Der Standard", "AT", "derstandard.at", "Wiener Tageszeitung, One Million Posts Corpus", "#B33A3A"),
        ("Süddeutsche Zeitung", "DE", "sueddeutsche.de", "Münchner Tageszeitung, MLSUM-Korpus", "#2F6690"),
    ],
    columns=["outlet", "country", "site", "profile", "colour"],
).set_index("outlet")

# Feste Farben je Zeitung, damit eine Zeitung in allen Grafiken gleich aussieht.
OUTLET_COLOURS = OUTLETS["colour"].to_dict()

# Die Ressorts der beiden Redaktionen heißen verschieden und sind verschieden
# geschnitten. Für Vergleiche "Gleiches mit Gleichem" ordnen wir die Ressorts mit
# einem klaren Gegenstück einer gemeinsamen Gruppe zu; alle übrigen bleiben ohne
# Gruppe (None).
SECTION_GROUPS = {
    "Der Standard": {
        "International": "Politik",
        "Inland": "Politik",
        "Wirtschaft": "Wirtschaft",
        "Sport": "Sport",
        "Panorama": "Panorama",
        "Web": "Digital",
    },
    "Süddeutsche Zeitung": {
        "politik": "Politik",
        "wirtschaft": "Wirtschaft",
        "geld": "Wirtschaft",
        "sport": "Sport",
        "panorama": "Panorama",
        "digital": "Digital",
    },
}

# ---------------------------------------------------------------------------
# Herunterladen und Umwandeln (einmalig)
# ---------------------------------------------------------------------------


def _download(url: str, target: Path) -> None:
    """Lade eine Datei herunter, zuerst unter einem temporären Namen."""
    if target.exists():
        return
    print(f"lade {url} ...", flush=True)
    tmp = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "textmining-course"})
    with urllib.request.urlopen(request) as response, open(tmp, "wb") as out:
        shutil.copyfileobj(response, out, length=1 << 20)
    tmp.rename(target)


def _prepare_standard(raw: Path) -> None:
    """One Million Posts Corpus -> eine Zeile je Der-Standard-Artikel (alle 12.087)."""
    archive = raw / "million_post_corpus.tar.bz2"
    _download(OMPC_URL, archive)
    db = raw / "million_post_corpus" / "corpus.sqlite3"
    if not db.exists():
        with tarfile.open(archive, "r:bz2") as tar:
            member = tar.getmember("million_post_corpus/corpus.sqlite3")
            tar.extractall(raw, members=[member], filter="data")
    with sqlite3.connect(db) as con:
        articles = pd.read_sql("SELECT ID_Article, Path, publishingDate, Title, Body FROM Articles", con)
    pd.DataFrame(
        {
            "article_id": articles["ID_Article"],
            "published": pd.to_datetime(articles["publishingDate"]),
            "path": articles["Path"],
            "title": articles["Title"],
            "body_html": articles["Body"].fillna(""),
        }
    ).to_parquet(STANDARD_FILE, index=False)


def _prepare_sz(raw: Path) -> None:
    """MLSUM Deutsch -> eine Zeile je SZ-Artikel aus dem gemeinsamen Zeitraum."""
    import pyarrow.parquet as pq

    columns = ["url", "date", "topic", "title", "target", "text"]
    parts = []
    for i, url in enumerate(MLSUM_URLS):
        path = raw / f"mlsum_de_train_{i}.parquet"
        _download(url, path)
        part = pq.read_table(path, columns=columns).to_pandas()
        # MLSUM kennt nur den Monat: "00/06/2015" steht für Juni 2015.
        month = pd.to_datetime("01/" + part["date"].str[3:], format="%d/%m/%Y", errors="coerce")
        parts.append(part.assign(month=month)[(month >= START) & (month < END)])
    sz = pd.concat(parts, ignore_index=True)
    pd.DataFrame(
        {
            "article_id": sz["url"].str.extract(r"-(1\.\d+)")[0],
            "month": sz["month"],
            "url": sz["url"],
            "topic": sz["topic"],
            "title": sz["title"],
            "teaser": sz["target"],
            "body": sz["text"],
        }
    ).to_parquet(SZ_FILE, index=False)


def prepare_data(keep_raw: bool = False) -> None:
    """Lade die Rohdaten herunter und wandle sie in `derstandard.parquet` und `sueddeutsche.parquet` um.

    Der Download umfasst rund 1 GB (das One Million Posts Corpus als SQLite-Datenbank
    und den deutschen Trainingsteil von MLSUM). Behalten werden nur die Artikel; die
    Leserkommentare des One Million Posts Corpus und alle MLSUM-Artikel außerhalb
    des gemeinsamen Zeitraums fallen weg. Die Rohdateien werden danach gelöscht,
    außer `keep_raw=True`.
    """
    raw = DATA / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    if not STANDARD_FILE.exists():
        _prepare_standard(raw)
    if not SZ_FILE.exists():
        _prepare_sz(raw)
    if not keep_raw:
        shutil.rmtree(raw, ignore_errors=True)


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Die beiden Quellen, so wie sie geliefert werden (vor allen Entscheidungen aus Skript 01)."""
    if not (STANDARD_FILE.exists() and SZ_FILE.exists()):
        prepare_data()
    return pd.read_parquet(STANDARD_FILE), pd.read_parquet(SZ_FILE)


# ---------------------------------------------------------------------------
# Bereinigung
# ---------------------------------------------------------------------------

_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")
_H2_RE = re.compile(r'<h2 itemprop="description">(.*?)</h2>', re.S)
_P_RE = re.compile(r"<p>(.*?)</p>", re.S)


def clean_html(text: str | None) -> str:
    """Entferne Markup und HTML-Entities.

    Der Standard liefert seine Artikel als HTML. Bliebe das Markup im Text, würden
    Tags, Link-Adressen und Entities wie `&amp;` zu sinnlosen "Wörtern" zerlegt.
    """
    if not text:
        return ""
    text = _TAG_RE.sub(" ", text)
    text = html.unescape(text)
    return _SPACE_RE.sub(" ", text).strip()


def standard_lead(body_html: str) -> str:
    """Der Teaser eines Standard-Artikels: der Vorspann (<h2>), sonst der erste Absatz."""
    match = _H2_RE.search(body_html) or _P_RE.search(body_html)
    return clean_html(match.group(1)) if match else ""


def undo_csv_quotes(title: str) -> str:
    r'''Mache das CSV-Escaping rückgängig, das MLSUM in SZ-Titeln mit Anführungszeichen hinterlassen hat.

    '"Obama singt ""Amazing Grace"""' -> 'Obama singt "Amazing Grace"'
    '''
    if len(title) >= 2 and title.startswith('"') and title.endswith('"') and '""' in title:
        return title[1:-1].replace('""', '"')
    return title


def standard_section(path: str) -> str:
    """Das Ressort aus dem Pfad: 'Newsroom/Inland/bundeslaender/...' -> 'Inland'."""
    parts = path.split("/")
    return parts[1] if len(parts) > 1 else parts[0]


# Ein Titel, den eine Zeitung mindestens so oft vergibt, benennt ein Format oder
# eine Rubrik ("Sport in der Region", "SZ Espresso ..."), keine Meldung.
FORMAT_MIN_REPEATS = 3


def load_corpus(dedupe: bool = True) -> pd.DataFrame:
    """Lade eine Zeile je Artikel, fertig für das Text Mining.

    Ein Dokument ist eine Schlagzeile (`title`); `text` ist Schlagzeile plus Teaser.
    Die Entscheidungen, Schritt für Schritt erklärt in Skript 01:

    1. Gemeinsamer Zeitraum: Juni 2015 bis Mai 2016. Das One Million Posts Corpus
       enthält einige ältere Artikel, die nur wegen neuer Leserkommentare darin sind.
    2. Nur Journalismus: Die Community-Seiten von Der Standard (Ressort `User`:
       Forenregeln, Leserfragen, Off-Topic-Foren) und die Gewinnspiel- und
       Themenseiten der SZ (`service`, `thema`) sind keine Artikel.
    3. Platzhalter statt Schlagzeile: Bei einigen hundert SZ-Artikeln (meist aus der
       Printausgabe) steht als Titel nur "Süddeutsche.de".
    4. Formate statt Meldungen: Titel, die eine Zeitung mindestens
       FORMAT_MIN_REPEATS-mal vergibt, benennen eine Rubrik ("Sport in der Region",
       "SZ Espresso ...", "Was haben Sie in den letzten Wochen gelesen?").
    5. Doppelte Titel (optional, `dedupe`): Die erste Fassung wird behalten.
    6. HTML: Der Standard liefert HTML; Teaser und Titel werden davon befreit.
    7. Anführungszeichen: In SZ-Titeln mit Anführungszeichen hat der CSV-Export
       von MLSUM Spuren hinterlassen (doppelte Anführungszeichen); sie
       werden rückgängig gemacht.
    8. Teaser: Bei Der Standard der Vorspann des Artikels, bei der SZ die von der
       Redaktion geschriebene Zusammenfassung aus MLSUM. Der erste Absatz des
       SZ-Textes taugt nicht, weil er oft eine Bildunterschrift ist.
    """
    standard, sz = load_raw()

    # Der Standard: Zeitraum, Ressort, Teaser
    standard = standard[(standard["published"] >= START) & (standard["published"] < END)]
    section = standard["path"].map(standard_section)
    at = pd.DataFrame(
        {
            "doc_key": "std-" + standard["article_id"].astype(str),
            "outlet": "Der Standard",
            "title": standard["title"].map(clean_html),
            "lead": standard["body_html"].map(standard_lead),
            "published": standard["published"],
            "month": standard["published"].dt.to_period("M").dt.to_timestamp(),
            "section": section,
            "path": standard["path"],
        }
    )[section != "User"]

    # Süddeutsche Zeitung: Ressort aus der URL, Platzhaltertitel, Teaser
    topic = sz["url"].str.extract(r"sueddeutsche\.de/([^/]+)/", expand=False)
    de = pd.DataFrame(
        {
            "doc_key": "sz-" + sz["article_id"].astype(str),
            "outlet": "Süddeutsche Zeitung",
            "title": sz["title"].str.strip().map(undo_csv_quotes),
            "lead": sz["teaser"].fillna("").str.strip(),
            "published": pd.NaT,  # MLSUM kennt nur den Monat
            "month": sz["month"],
            "section": topic,
            "path": sz["url"],
        }
    )
    de = de[~de["section"].isin(["service", "thema"]) & (de["title"] != "Süddeutsche.de")]

    corpus = pd.concat([at, de], ignore_index=True)
    corpus["country"] = corpus["outlet"].map(OUTLETS["country"])
    corpus["section_group"] = [SECTION_GROUPS[o].get(s) for o, s in zip(corpus["outlet"], corpus["section"])]

    # Formate: Titel, die eine Zeitung mehrfach vergibt
    repeats = corpus.groupby(["outlet", "title"])["title"].transform("size")
    corpus = corpus[repeats < FORMAT_MIN_REPEATS]

    # Schlagzeile und Teaser gemeinsam, für Skripte, die mehr Text je Dokument
    # brauchen. Der Punkt dazwischen verhindert, dass das letzte Wort des Titels und
    # das erste Wort des Teasers ein falsches Bigramm bilden.
    corpus["text"] = corpus["title"].where(corpus["lead"] == "", corpus["title"] + ". " + corpus["lead"])

    if dedupe:
        corpus = corpus.sort_values(["month", "published"]).drop_duplicates(["outlet", "title"], keep="first")

    return corpus.sort_values(["outlet", "month", "published"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Die Klassifikationsaufgabe der Skripte 05-07
# ---------------------------------------------------------------------------

# Die fünf größten Ressorts von Der Standard werden zu Klassen; alle übrigen
# (Inland, Etat, Wissenschaft, Kultur, Meinung, Lifestyle, ...) werden zu
# "Sonstiges" zusammengefasst.
SUBJECTS = ["International", "Panorama", "Web", "Wirtschaft", "Sport"]
OTHER = "Sonstiges"


def subject_task(test_share: float = 0.25) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Die Klassifikationsaufgabe der Skripte 05-07: Sage das Standard-Ressort aus der Schlagzeile vorher.

    Die Aufteilung erfolgt zeitlich, nicht zufällig: Der jüngste Anteil
    `test_share` der Artikel wird zurückgehalten. Nachrichten wiederholen sich -
    Folgemeldungen, Korrekturen, fast identische Schlagzeilen an aufeinanderfolgenden
    Tagen -, sodass eine zufällige Aufteilung Beinahe-Kopien der Testdaten in die
    Trainingsdaten bringen und die Genauigkeit überschätzen würde. Alle Skripte
    verwenden diese eine Aufteilung; nur deshalb sind ihre Ergebnisse vergleichbar.
    """
    corpus = load_corpus()
    std = corpus[corpus["outlet"] == "Der Standard"].copy()
    std["label"] = std["section"].where(std["section"].isin(SUBJECTS), OTHER)
    std = std.sort_values("published").reset_index(drop=True)
    cut = int(len(std) * (1 - test_share))
    return std.iloc[:cut].reset_index(drop=True), std.iloc[cut:].reset_index(drop=True)


# ---------------------------------------------------------------------------
# spaCy-Cache
# ---------------------------------------------------------------------------


def parse_corpus(nlp, texts: list[str], name: str, batch_size: int = 256, n_process: int = 1):
    """Lass spaCy einmal über die Texte laufen und speichere das Ergebnis auf der Festplatte.

    Das Parsen zehntausender Schlagzeilen dauert lange genug, um zu stören, wenn es
    in jedem Skript wiederholt wird. DocBin ist das Binärformat von spaCy für genau
    diesen Zweck. Der Cache-Schlüssel enthält den Modellnamen und die Anzahl der
    Texte; ein anderer Korpus oder ein anderes Modell führt daher zu einem neuen
    Parse statt zu einem veralteten.
    """
    from spacy.tokens import DocBin

    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"{name}-{nlp.meta['name']}-{len(texts)}.spacy"
    if path.exists():
        docs = list(DocBin().from_disk(path).get_docs(nlp.vocab))
        if [d.text for d in docs] == list(texts):
            return docs
    docs = list(nlp.pipe(texts, batch_size=batch_size, n_process=n_process))
    DocBin(store_user_data=False, docs=docs).to_disk(path)
    return docs
