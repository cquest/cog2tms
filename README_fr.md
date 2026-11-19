# cog2tms

[![AI:W WRITE](https://img.shields.io/badge/AI-WRITE-grey?style=flat&logoSize=auto&labelColor=13171f)](https://0rwx.com)
[![ci](https://github.com/cquest/cog2tms/actions/workflows/ci.yml/badge.svg)](https://github.com/cquest/cog2tms/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/cquest/cog2tms/graph/badge.svg)](https://codecov.io/gh/cquest/cog2tms)
![license: WTFPL](https://img.shields.io/badge/license-WTFPL-green)
![python: 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)

[English](README.md)

Donnez-lui l'URL d'un GeoTIFF optimisé pour le cloud (COG) : il lit le fichier à
distance (uniquement des gammes d'octets HTTP, jamais le fichier entier) et
servit des tuiles Z/X/Y (schéma OSM) reprojetées en EPSG:3857 (Web Mercator).

## API

```
GET /                                       # interface web : aperçu MapLibre (fond vectoriel) + URL de tuiles
GET /c2t/{z}/{x}/{y}?u=<url du COG>[&format=png|jpeg|webp|auto][&quality=80][&size=512]
                      [&bidx=1,2,3][&rescale=min,max][&blackmax=0]
GET /c2t/{z}/{x}/{y}.geojson?u=<url du COG> # empreinte du COG (debug)
GET /c2t.tms.json?u=<url du COG>            # TileJSON (bbox, minzoom/maxzoom)
GET /healthz   GET /robots.txt   GET /static/…   # santé, politique robots, ressources de l'interface
GET /openapi.json   GET /docs   GET /redoc       # schéma OpenAPI 3.1 + docs (swagger-ui via CDN)
# toute route ci-dessus répond aussi à HEAD : mêmes en-têtes, pas de corps
```

- Numérotation XYZ (OSM/Leaflet, y vers le bas). Formats : PNG (RGBA), JPEG,
  WebP (≈12× plus petit que le PNG sur des tuiles d'orthophoto). `format=auto`
  ne fait confiance qu'à l'en-tête `Accept` : webp si `image/webp` est listé
  explicitement (q>0), jamais sur les jokers (`*/*` — JOSM les envoie et ne sait
  pas décoder le webp) ; défaut png.
- Taille de tuile par défaut `512`, `size=128..1024` restreint aux multiples de
  `16` (macrobloc JPEG / bloc TIFF)
- Tuiles hors de l'empreinte du COG : `200` + tuile vide — PNG/WebP
  transparents, JPEG en remplissage `EMPTY_TILE_BG`.
- `blackmax=0..255` : les pixels quasi noirs (lisiérés d'orthophotos sans
  masque) passent en non-couverture ; défaut `20`, `-1` désactive.
- transparence sur NODATA
- Erreurs : `400` zoom / indice / format / taille / URL, données signées, 32
  bits ou flottantes sans `rescale`, SIG non pris en charge, chaîne de
  redirections refusée ; `404` COG injoignable, pas un GeoTIFF ou format
  d'échantillons inutilisable ; `502` erreur en amont.

## Lancer en local

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
uvicorn app.main:app --reload --port 3857
```

L'interface `http://127.0.0.1:3857/` prend l'URL d'un COG, affiche un aperçu
MapLibre et des URL de tuiles prêtes à coller. En ligne de commande (exemple
public d'orthophoto, EPSG:3943) :

```bash
curl -o tile.png \
  "http://127.0.0.1:3857/c2t/18/130034/95856?u=https://data.cquest.org/temp/cog4josm/COG_BayonneSudEst.tif"
```

Dans JOSM (Paramètres d'affichage de la carte → sources WMS/TMS → nouvelle
source `TMS` — son `{y}` est déjà haut→bas) :

```
tms:http://127.0.0.1:3857/c2t/{zoom}/{x}/{y}?u=https://data.cquest.org/temp/cog4josm/COG_BayonneSudEst.tif
```

## Tests

```bash
pytest              # hors ligne : COGs de test écrits en interne et servis localement
pytest -m network   # intégration sur les COGs publics cog4josm
pytest --cov        # couverture de branches de app/ (seuil imposé, voir pyproject)
```

## Performance

Mesuré sur le COG de Bayonne (875 Mo, EPSG:3943) depuis un poste de travail :

| Opération | Mesuré |
|---|---|
| ouverture du COG (en-têtes TIFF/GeoTIFF) | 0,05 s, 1 req, 64 Ko |
| première tuile, processus neuf (ouverture + rendu) | ≈0,15–0,3 s |
| première tuile au zoom max (index de blocs L0) | pas de coût supplémentaire |
| rendu à chaud (blocs en cache) | ≈6 ms |
| répétition de la même tuile (cache de tuiles) | ≈2 ms |
| encodage d'une tuile z18 (256 px) | PNG 134 Ko, WebP 11,4 Ko (≈12× plus petit, ≈2 ms), JPEG 13,5 Ko |

Les leviers derrière ces chiffres (préchargement d'en-tête, index de blocs
paresseux, requêtes fusionnées/parallèles/multi-gammes, warp `cv2.remap`,
décodage threadé, LRU de tuiles avec verrous single-flight) :
[AGENTS.md](AGENTS.md).

## Configuration (variables d'environnement)

| Variable | Défaut | Signification |
|---|---|---|
| `MAX_ZOOM` | `24` | niveau de zoom le plus élevé servi |
| `READER_TTL` | `600` | secondes pendant lesquelles un COG ouvert reste en cache |
| `POOL_SIZE` | `64` | maximum de COGs ouverts simultanément |
| `MAX_CONCURRENT_READS` | `8` | maximum de rendus de tuiles simultanés |
| `BLOCK_CACHE` | `1500` | blocs COG décodés conservés par COG ouvert |
| `DECODE_THREADS` | `4` | threads pour décoder les blocs d'une grande fenêtre (`1` : décodage en ligne) |
| `TILE_CACHE` | `1024` | tuiles encodées gardées en mémoire, LRU (`0` désactive) |
| `PNG_COMPRESS` | `1` | niveau deflate PNG (1 est ~4× plus rapide que 6, ~13 % plus gros) |
| `WEBP_LOSSLESS` | `0` | `1` : WebP sans perte+exact (identique bit à bit au PNG, ~2× la taille avec perte) |
| `MERGE_GAP` | `4096` | deux gammes plus proches que cela partagent une requête HTTP |
| `MULTIRANGE` | `1` | plusieurs gammes par GET (multipart/byteranges) ; le refus est mémorisé par hôte |
| `HEAD_PREFETCH` | `65536` | lecture en bloc de l'en-tête (octets) absorbant l'analyse IFD/balises à l'ouverture (`0` désactive) |
| `HEAD_CACHE` | — | fichier sqlite conservant ces plages d'en-tête après un redémarrage (COG avec ETag/Last-Modified uniquement, lignes bornées 4096 + TTL 30 j) ; vide désactive |
| `HTTP_CONNECT_TIMEOUT` / `HTTP_READ_TIMEOUT` | `10` / `30` | délais d'attente amont (s) |
| `HTTP_RETRIES` / `HTTP_BACKOFF` | `2` / `0,3` | retries transparents sur 429/5xx (`0` désactive) |
| `HTTP2` | `1` | HTTP/2 vers les hôtes qui le proposent (ALPN) ; `0` épingle HTTP/1.1 |
| `NEGATIVE_TTL` | `30` | un 404/403 à distance court-circuite ce COG pendant cette durée (`0` désactive) |
| `VALIDATE_TTL` | `30` | secondes entre deux vérifications de version `If-Range` (`-1` désactive) |
| `VALIDATE_TIMEOUT` | `5` | délai d'attente (s) de cette vérification de version |
| `MAX_REDIRECTS` | `5` | sauts 30x suivis, chacun revalidé (`0` refuse les redirections) |
| `EMPTY_TILE_BG` | `0,0,0` | RGB plein utilisé pour les tuiles JPEG vides |
| `BLACKMAX` | `20` | seuil de transparence quasi noir par défaut (`-1` désactive) |
| `SHARPEN` | `0,3` | masque de netteté sur les tuiles issues d'overviews (tuiles pleine résolution intactes ; `0` désactive) |
| `ALLOWED_HOSTS` | — | si défini, liste blanche d'hôtes COG séparée par des virgules |

- **Anti-SSRF** : http(s) uniquement ; IPs privées/réservées rejetées
  (dev/tests : `ALLOW_PRIVATE_NET=1`) ; chaque `Location` de redirection est
  revalidée avant connexion ; la politique d'URL (liste blanche incluse)
  s'applique à chaque requête, cache compris.
- **Résilience** : 429/5xx retryés en transparent (`Retry-After` respecté) ;
  404/403 à distance mémorisés ; les tuiles en cache continuent d'être servies
  tant que l'amont est en panne.
- **Cohérence** : l'`ETag`/`Last-Modified`/taille de chaque COG est ancré et les
  requêtes suivantes portent `If-Range` ; toute dérive fait tomber le lecteur
  et ses tuiles en cache — la réponse 200 fichier entier d'un remplacement n'est
  jamais téléchargée. Avec `HEAD_CACHE`, les plages d'en-tête survivent à un
  redémarrage uniquement pour les COG dont les réponses portent un tel
  validateur : la réouverture ré-ancre `If-Range` depuis le validateur stocké,
  un fichier remplacé pendant l'arrêt est donc détecté dès la première lecture
  de bloc (et les lignes périmées purgées).

## Docker

```bash
docker build -t cog2tms .
docker run -p 3857:3857 cog2tms   # http://localhost:3857
# plages d'en-tête conservées dans un volume après redémarrage (COG avec ETag/Last-Modified) :
docker run -p 3857:3857 -v cog2tms-cache:/var/cache/cog2tms \
  -e HEAD_CACHE=/var/cache/cog2tms/heads.sqlite3 cog2tms
```

## Licence

WTFPL v2 — voir [LICENSE](LICENSE). Les fichiers tiers vendorisés gardent leurs
propres termes : MapLibre GL JS (BSD-3-Clause, non modifié), icône JOSM
(domaine public).
