# cog2tms

[![AI:W WRITE](https://img.shields.io/badge/AI-WRITE-grey?style=flat&logoSize=auto&labelColor=13171f)](https://0rwx.com)
[![ci](https://github.com/cquest/cog2tms/actions/workflows/ci.yml/badge.svg)](https://github.com/cquest/cog2tms/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/cquest/cog2tms/graph/badge.svg)](https://codecov.io/gh/cquest/cog2tms)
![license: WTFPL](https://img.shields.io/badge/license-WTFPL-green)
![python: 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)

[English](README.md)

Service de tuiles cartographiques en TMS Z/X/Y depuis un GeoTIFF optimisé pour le cloud
(COG).

Donnez-lui l'URL d'un COG : il lit le fichier à distance et
renvoie des tuiles Z/X/Y (schéma OSM) reprojetées en EPSG:3857 (Web Mercator).

## API

```
GET /                                        # interface web (formulaire + aperçu MapLibre, fond vectoriel)
GET /static/…                                # ressources de l'interface (MapLibre vendorisé, maplibre_app.js)
GET /c2t/{z}/{x}/{y}?u=<url du COG>[&format=png|jpeg|webp|auto][&quality=80][&size=256]
                      [&bidx=1,2,3][&rescale=min,max][&blackmax=0]
GET /c2t/{z}/{x}/{y}.geojson?u=<url du COG>  # emprise du COG (debug)
GET /c2t.tms.json?u=<url du COG>             # TileJSON (bbox, minzoom/maxzoom)
GET /healthz
GET /robots.txt
```

- Schéma de tuiles : XYZ (numérotation OSM/Leaflet, y vers le bas). Sortie PNG
  (RGBA), JPEG (`EMPTY_TILE_BG` plein en dehors de la couverture) ou WebP
  (≈13× plus petit que le PNG sur des tuiles d'orthophoto ; `WEBP_LOSSLESS=1`
  pour du WebP sans perte).
- `format=auto` ne suit que l'en-tête `Accept` : webp si `image/webp` est listé
  explicitement (q>0), sinon png. Les jokers (`image/*`, `*/*`) ne sélectionnent
  jamais webp — JOSM les envoie et ne sait pas le décoder.
- `blackmax=0..255` : les pixels dont la bande la plus lumineuse est ≤ cette
  valeur (quasi noirs — typiquement les franges non masquées en bord
  d'orthophoto) deviennent « hors couverture ». Défaut `20` ; `-1` désactive.
- Tuiles hors de l'emprise du COG : `200` avec une tuile vide (transparente).
- Erreurs : `400` zoom / indice / format / taille / URL invalides, données
  16 bits sans `rescale`, SCR non pris en charge, chaîne de redirections
  refusée ; `404` COG injoignable ou pas un GeoTIFF ; `502` erreur en amont.
- Anti-extraction : `robots.txt` interdit l'API de tuiles (`/c2t`) et chaque
  réponse porte `X-Robots-Tag: nofollow, noimageindex` — les URLs de tuiles
  sont des images que robots.txt seul ne couvrirait pas. (Dissuade les bons
  robots, pas les scrapers.)

## Lancer en local

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
uvicorn app.main:app --reload --port 3857
```

L'interface web sur `http://127.0.0.1:3857/` : collez une URL de COG, obtenez
un aperçu MapLibre et des URLs de tuiles prêtes à coller. Ou directement depuis
le shell (le COG d'exemple est une orthophoto drone publique, EPSG:3943) :

```bash
curl -o tile.png \
  "http://127.0.0.1:3857/c2t/18/130034/95856?u=https://data.cquest.org/temp/cog4josm/COG_BayonneSudEst.tif"
```

Dans JOSM (Réglages d'affichage de la carte → sources WMS/TMS → nouvelle source
`TMS`) — le `{y}` de JOSM est déjà de haut en bas, pas d'inversion nécessaire :

```
tms:http://127.0.0.1:3857/c2t/{zoom}/{x}/{y}?u=https://data.cquest.org/temp/cog4josm/COG_BayonneSudEst.tif
```

## Tests

```bash
pytest              # hors ligne : COGs de fixture écrits en interne et servis localement
pytest -m network   # intégration sur les COGs publics cog4josm
pytest --cov        # couverture de branches de app/ (seuil appliqué, voir pyproject)
```

## Performance

Mesuré sur le COG de Bayonne (875 Mo, EPSG:3943) depuis un poste de travail :

| Opération | Mesuré |
|---|---|
| ouverture du COG (en-têtes TIFF/GeoTIFF) | 0,05 s, 1 requête, 64 Ko |
| première tuile, processus neuf (ouverture + rendu) | ≈0,15–0,3 s |
| première tuile au zoom max (index de blocs L0) | surcoût nul |
| rendu à chaud (blocs en cache) | ≈6 ms |
| répétition de la même tuile (cache de tuiles) | ≈2 ms |
| encodage d'une tuile z18 | PNG 103 Ko, WebP 7,9 Ko (≈13× plus petit, ≈2 ms), JPEG 10 Ko |

Leviers (préchargement en bloc de l'en-tête, index de blocs paresseux,
allègements fusionnés/parallèles/multi-range, warp `cv2.remap`, décodage
multi-thread, cache LRU de tuiles avec verrous single-flight) : détails dans
[AGENTS.md](AGENTS.md).

## Configuration (variables d'environnement)

| Variable | Défaut | Signification |
|---|---|---|
| `MAX_ZOOM` | `24` | niveau de zoom maximal servi |
| `READER_TTL` | `600` | secondes pendant lesquelles un COG ouvert reste en cache |
| `POOL_SIZE` | `64` | maximum de COGs ouverts simultanément |
| `MAX_CONCURRENT_READS` | `8` | maximum de rendus de tuiles simultanés |
| `BLOCK_CACHE` | `1500` | blocs COG décodés conservés par COG ouvert |
| `DECODE_THREADS` | `4` | threads décodant les blocs d'une grande fenêtre (`1` décode en ligne) |
| `TILE_CACHE` | `1024` | tuiles encodées gardées en mémoire, LRU (`0` désactive) |
| `PNG_COMPRESS` | `1` | niveau deflate PNG (1 est ~4× plus rapide que 6, ~13 % plus gros) |
| `WEBP_LOSSLESS` | `0` | `1` : WebP sans perte+exact (identique au PNG bit à bit, ~2× la taille du lossy) |
| `MERGE_GAP` | `4096` | les byte-ranges plus proches que cela partagent une requête HTTP |
| `MULTIRANGE` | `1` | plusieurs ranges par GET (multipart/byteranges) ; le refus est mémorisé par hôte |
| `HEAD_PREFETCH` | `65536` | lecture en bloc de l'en-tête (octets) absorbant l'analyse IFD/balises à l'ouverture (`0` désactive) |
| `HTTP_CONNECT_TIMEOUT` / `HTTP_READ_TIMEOUT` | `10` / `30` | délais amont (s) |
| `HTTP_RETRIES` / `HTTP_BACKOFF` | `2` / `0,3` | retries transparents sur 429/5xx (`0` désactive) |
| `NEGATIVE_TTL` | `30` | un 404/403 amont court-circuite ce COG pendant cette durée (`0` désactive) |
| `VALIDATE_TTL` | `30` | secondes entre deux vérifications de version `If-Range` (`-1` désactive) |
| `VALIDATE_TIMEOUT` | `5` | délai (s) de cette vérification de version |
| `MAX_REDIRECTS` | `5` | sauts 30x suivis, chacun re-validé (`0` refuse les redirections) |
| `EMPTY_TILE_BG` | `0,0,0` | RVB plein utilisé pour les tuiles JPEG vides |
| `BLACKMAX` | `20` | seuil de transparence quasi noir par défaut (`-1` désactive) |
| `ALLOWED_HOSTS` | — | si définie, liste blanche d'hôtes COG séparés par virgules |

- **Anti-SSRF** : URLs de COG `http(s)` uniquement ; les cibles résolvant vers
  des IP privées ou réservées sont rejetées (dev/tests : `ALLOW_PRIVATE_NET=1`) ;
  chaque `Location` d'une chaîne 30x est re-validée avant connexion.
- **Résilience** : 429/5xx réessayés transparentement (`Retry-After` respecté) ;
  404/403 amont mémorisés par URL ; les tuiles en cache continuent d'être
  servies quand l'amont est en panne.
- **Cohérence** : l'`ETag`/`Last-Modified`/taille de chaque COG est ancré et les
  requêtes ultérieures portent `If-Range` ; toute dérive évince le lecteur de ce
  COG et ses tuiles en cache — la réponse 200 fichier entier d'un fichier
  remplacé n'est jamais téléchargée.

## Docker

```bash
docker build -t cog2tms .
docker run -p 3857:3857 cog2tms   # http://localhost:3857
```

Notes de conception, liste des SCR pris en charge et feuille de route :
[AGENTS.md](AGENTS.md).

## Licence

WTFPL v2 — voir [LICENSE](LICENSE). Les fichiers tiers vendorisés gardent leurs
propres termes : MapLibre GL JS (BSD-3-Clause, non modifié), l'icône JOSM
(domaine public).
