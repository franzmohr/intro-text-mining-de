"""Gemeinsame Hilfsfunktionen für die Text-Mining-Skripte des Kurses.

Alle Skripte arbeiten mit demselben Korpus: Schlagzeilen und Teaser zweier
Tageszeitungen aus demselben Jahr (Juni 2015 bis Mai 2016),

- **Der Standard** (Wien, AT) aus dem One Million Posts Corpus (Schabus et al. 2017,
  CC BY-NC-SA 4.0) und
- **Süddeutsche Zeitung** (München, DE) aus dem MLSUM-Korpus (Scialom et al. 2020,
  nur für nicht-kommerzielle Forschung und Lehre).

Die Entscheidungen darüber, was als Dokument zählt, stehen einmal hier - statt in
neun Notebooks wiederholt zu werden und dabei unbemerkt auseinanderzulaufen.
Skript 01 erklärt jede dieser Entscheidungen Schritt für Schritt; dieses Modul ist
die Kurzfassung. Dazu kommen die vollen Artikeltexte (`load_bodies()`) und die von
der Redaktion annotierten Leserkommentare von Der Standard (`load_posts()`).

Die Rohdaten werden beim ersten Aufruf von `load_corpus()` heruntergeladen und in
drei kleine Parquet-Dateien umgewandelt (`prepare_data()`, etwa 1 GB Download,
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
from contextlib import closing
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
POSTS_FILE = DATA / "derstandard_posts.parquet"
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
        expected = response.headers.get("Content-Length")
        shutil.copyfileobj(response, out, length=1 << 20)
    # Bricht die Verbindung vorzeitig ab, meldet urllib keinen Fehler; eine
    # abgeschnittene Datei würde erst beim Entpacken auffallen. Daher die Größe prüfen.
    if expected is not None and tmp.stat().st_size != int(expected):
        tmp.unlink()
        raise OSError(f"Download von {url} unvollständig, bitte erneut ausführen")
    tmp.replace(target)


def _ompc_database(raw: Path) -> Path:
    """Lade das One Million Posts Corpus herunter und entpacke seine SQLite-Datenbank."""
    archive = raw / "million_post_corpus.tar.bz2"
    _download(OMPC_URL, archive)
    db = raw / "million_post_corpus" / "corpus.sqlite3"
    if not db.exists():
        with tarfile.open(archive, "r:bz2") as tar:
            member = tar.getmember("million_post_corpus/corpus.sqlite3")
            tar.extractall(raw, members=[member], filter="data")
    return db


def _prepare_posts(raw: Path) -> None:
    """One Million Posts Corpus -> eine Zeile je annotiertem Leserkommentar.

    Die Redaktion von Der Standard hat rund 11.000 Kommentare nach neun Kategorien
    annotiert (Argumente, Stimmung, Off-Topic, persönliche Geschichten, ...). Jede
    Kategorie wird eine Spalte mit 1, 0 oder fehlend (nicht annotiert).
    """
    with closing(sqlite3.connect(_ompc_database(raw))) as con:
        posts = pd.read_sql(
            """SELECT p.ID_Post, p.ID_Article, p.CreatedAt, p.Headline, p.Body,
                      p.PositiveVotes, p.NegativeVotes, a.Path, a.Title AS ArticleTitle
               FROM Posts p JOIN Articles a USING (ID_Article)
               WHERE p.ID_Post IN (SELECT ID_Post FROM Annotations_consolidated)""",
            con,
        )
        labels = pd.read_sql("SELECT ID_Post, Category, Value FROM Annotations_consolidated", con)
    labels = labels.pivot(index="ID_Post", columns="Category", values="Value").astype("Int8")
    posts = posts.set_index("ID_Post").join(labels).reset_index()
    pd.DataFrame(
        {
            "post_id": posts["ID_Post"],
            "article_id": posts["ID_Article"],
            "created": pd.to_datetime(posts["CreatedAt"]),
            "headline": posts["Headline"].fillna("").str.strip(),
            "body": posts["Body"].fillna("").str.strip(),
            "upvotes": posts["PositiveVotes"],
            "downvotes": posts["NegativeVotes"],
            "article_title": posts["ArticleTitle"].map(clean_html),
            "article_section": posts["Path"].map(standard_section),
        }
    ).join(posts[list(labels.columns)]).to_parquet(POSTS_FILE, index=False)


def _prepare_standard(raw: Path) -> None:
    """One Million Posts Corpus -> eine Zeile je Der-Standard-Artikel (alle 12.087)."""
    with closing(sqlite3.connect(_ompc_database(raw))) as con:
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
    if not POSTS_FILE.exists():
        _prepare_posts(raw)
    if not SZ_FILE.exists():
        _prepare_sz(raw)
    if not keep_raw:
        shutil.rmtree(raw, ignore_errors=True)


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Die beiden Quellen, so wie sie geliefert werden (vor allen Entscheidungen aus Skript 01)."""
    if not (STANDARD_FILE.exists() and SZ_FILE.exists() and POSTS_FILE.exists()):
        prepare_data()
    return pd.read_parquet(STANDARD_FILE), pd.read_parquet(SZ_FILE)


def load_posts() -> pd.DataFrame:
    """Die annotierten Leserkommentare von Der Standard (One Million Posts Corpus).

    Eine Zeile je Kommentar; `text` ist Überschrift plus Text des Kommentars. Die
    Spalten `ArgumentsUsed`, `SentimentNegative`, ... enthalten die Annotation der
    Redaktion (1/0) oder <NA>, wenn der Kommentar für diese Kategorie nicht annotiert wurde.
    """
    if not POSTS_FILE.exists():
        prepare_data()
    posts = pd.read_parquet(POSTS_FILE)
    posts["text"] = posts["headline"].where(posts["body"] == "", posts["headline"] + " " + posts["body"]).str.strip()
    return posts[posts["text"] != ""].reset_index(drop=True)


def load_bodies() -> pd.Series:
    """Der volle Artikeltext beider Zeitungen, als Series mit `doc_key` als Index.

    `load_corpus()` enthält nur Schlagzeile und Teaser. Für das Training eigener
    Wortvektoren (Skript 06), für das Argument Mining in Meinungsartikeln (Skript 08)
    und für die Zusammenfassung (Skript 09) brauchen wir die ganzen Artikel. Bei
    Der Standard wird der Vorspann (<h2>) weggelassen, weil er schon der Teaser ist.
    """
    standard, sz = load_raw()
    body_std = standard["body_html"].str.replace(_H2_RE, " ", regex=True).map(clean_html)
    # Einige Werbeseiten der SZ (Newsletter) haben keine Artikelnummer; sie sind keine Artikel.
    sz = sz[sz["article_id"].notna()]
    return pd.concat([
        pd.Series(body_std.to_numpy(), index="std-" + standard["article_id"].astype(str)),
        pd.Series(sz["body"].fillna("").str.strip().to_numpy(), index="sz-" + sz["article_id"].astype(str)),
    ]).rename("body")


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
# API-Schlüssel für die Skripte mit LLMs (07 und 09)
# ---------------------------------------------------------------------------

API_KEYS_FILE = HERE / "api-keys.env"


def load_api_keys(path: Path = API_KEYS_FILE) -> dict[str, bool]:
    """Lies die API-Schlüssel aus `api-keys.env` in die Umgebungsvariablen.

    Die Datei enthält Zeilen der Form `NAME=wert`; Kommentarzeilen beginnen mit `#`.
    Ein Eintrag in der Datei hat Vorrang vor einer gleichnamigen Umgebungsvariablen:
    Wird ein (zeitlich befristeter) Schlüssel in der Datei ausgetauscht, genügt es,
    die Funktion erneut aufzurufen - ein Neustart des Kernels ist nicht nötig. Leere
    Einträge werden übersprungen; dann gilt eine vorhandene Umgebungsvariable.
    Zurückgegeben wird, welche Schlüssel danach vorhanden sind - die Schlüssel selbst
    werden nie ausgegeben.
    """
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            name, sep, value = line.partition("=")
            name, value = name.strip(), value.strip().strip('"').strip("'")
            if sep and value and not name.startswith("#"):
                os.environ[name] = value
    return {name: bool(os.environ.get(name)) for name in ["ANTHROPIC_API_KEY", "MISTRAL_API_KEY"]}


# ---------------------------------------------------------------------------
# Die Klassifikationsaufgabe der Skripte 05-07
# ---------------------------------------------------------------------------

# Die sechs größten Ressorts von Der Standard werden zu Klassen; alle übrigen
# (Etat, Wissenschaft, Kultur, Meinung, Gesundheit, Lifestyle, ...) werden zu
# "Sonstiges" zusammengefasst.
SUBJECTS = ["International", "Inland", "Panorama", "Web", "Wirtschaft", "Sport"]
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
